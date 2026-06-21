use rustfft::num_complex::Complex;
use crate::fft2d::{fft2d, ifft2d};
use crate::webgpu::{ControlPointMeta, EdgeNodes};


#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct Node {
    pub x: f32,
    pub y: f32,
    pub mass: f32,
    #[serde(default)]
    pub degree: f32,
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
}

pub struct GravitySimulation {
    pub width: usize,
    pub height: usize,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub control_point_spacing: f32,
    pub potential_field: Vec<f32>,
    pub force_field_x: Vec<f32>,
    pub force_field_y: Vec<f32>,
    pub control_points: Vec<f32>, // Flat array: all coordinates
    pub control_point_offsets: Vec<usize>, // Starting index in control_points for each edge
    pub control_point_counts: Vec<usize>, // Number of control points for each edge
}

impl GravitySimulation {
    pub fn new(
        width: usize,
        height: usize,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        control_point_spacing: f32,
    ) -> Self {
        let mut sim = GravitySimulation {
            width,
            height,
            nodes,
            edges,
            control_point_spacing,
            potential_field: vec![0.0; width * height],
            force_field_x: vec![0.0; width * height],
            force_field_y: vec![0.0; width * height],
            control_points: Vec::new(),
            control_point_offsets: Vec::new(),
            control_point_counts: Vec::new(),
        };
        sim.reset_control_points();
        sim
    }

    pub fn update_nodes(&mut self, nodes: Vec<Node>) {
        self.nodes = nodes;
    }

    pub fn reset_control_points(&mut self) {
        let mut control_points = Vec::new();
        let mut control_point_offsets = Vec::with_capacity(self.edges.len());
        let mut control_point_counts = Vec::with_capacity(self.edges.len());
        
        let spacing = self.control_point_spacing.max(1.0);
        
        for edge in &self.edges {
            let start_node = &self.nodes[edge.source];
            let end_node = &self.nodes[edge.target];
            
            let dx = end_node.x - start_node.x;
            let dy = end_node.y - start_node.y;
            let dist = (dx * dx + dy * dy).sqrt();
            
            let n = ((dist / spacing).floor() as usize + 2).max(3);
            
            control_point_offsets.push(control_points.len() / 2);
            control_point_counts.push(n);
            
            for i in 0..n {
                let t = i as f32 / (n - 1) as f32;
                let x = start_node.x + t * dx;
                let y = start_node.y + t * dy;
                control_points.push(x);
                control_points.push(y);
            }
        }
        
        self.control_points = control_points;
        self.control_point_offsets = control_point_offsets;
        self.control_point_counts = control_point_counts;
    }

    pub fn update_physics_fields(&mut self, gravity_param: f32, potential_max: f32, gravity_alpha: f32) {
        // Compute resolution-independent scaling factor relative to baseline of 256
        let scale = self.width as f32 / 256.0;
        
        let potential_max_scaled = (potential_max / scale).max(1e-5);
        
        let mut potential = vec![0.0; self.width * self.height];
        let mut fx = vec![0.0; self.width * self.height];
        let mut fy = vec![0.0; self.width * self.height];
        
        for y in 0..self.height {
            for x in 0..self.width {
                let px = x as f32;
                let py = y as f32;
                
                let mut pot_sum = 0.0;
                let mut fx_sum = 0.0;
                let mut fy_sum = 0.0;
                
                for node in &self.nodes {
                    let dx = px - node.x;
                    let dy = py - node.y;
                    let d = (dx * dx + dy * dy).sqrt();
                    let softening_scaled = (gravity_param * node.mass) / potential_max_scaled;
                    let denom = (d - gravity_alpha * node.mass).max(softening_scaled);
                    
                    pot_sum -= (gravity_param * node.mass) / denom;
                    
                    if d > 0.0 {
                        if d - gravity_alpha * node.mass > softening_scaled {
                            let f_mag = (gravity_param * node.mass) / (denom * denom);
                            fx_sum += f_mag * (dx / d);
                            fy_sum += f_mag * (dy / d);
                        }
                    }
                }
                
                potential[y * self.width + x] = pot_sum * scale;
                fx[y * self.width + x] = fx_sum * scale * scale;
                fy[y * self.width + x] = fy_sum * scale * scale;
            }
        }
        
        self.potential_field = potential;
        self.force_field_x = fx;
        self.force_field_y = fy;
    }

    pub fn step(&mut self, spring_k: f32, dt: f32, damping: f32) {
        let num_edges = self.edges.len();
        let current_pts = self.control_points.clone();
        
        for e_idx in 0..num_edges {
            let offset = self.control_point_offsets[e_idx];
            let count = self.control_point_counts[e_idx];
            
            for i in 1..(count - 1) {
                let curr_base = 2 * (offset + i);
                let prev_base = 2 * (offset + i - 1);
                let next_base = 2 * (offset + i + 1);
                
                let px = current_pts[curr_base];
                let py = current_pts[curr_base + 1];
                
                // Hook's law for spring force between adjacent control points
                let f_spring_x = spring_k * (current_pts[prev_base] + current_pts[next_base] - 2.0 * px);
                let f_spring_y = spring_k * (current_pts[prev_base + 1] + current_pts[next_base + 1] - 2.0 * py);
                
                // Gravity pull from force vector fields
                let (f_grav_x, f_grav_y) = bilinear_interpolate_force(
                    self.width,
                    self.height,
                    &self.force_field_x,
                    &self.force_field_y,
                    px,
                    py,
                );
                
                let fx = f_spring_x + f_grav_x;
                let fy = f_spring_y + f_grav_y;
                
                let mut dx = fx * dt;
                let mut dy = fy * dt;
                
                // Clamp displacement to prevent instability
                let max_disp = 5.0;
                let disp_len = (dx * dx + dy * dy).sqrt();
                if disp_len > max_disp {
                    dx = (dx / disp_len) * max_disp;
                    dy = (dy / disp_len) * max_disp;
                }
                
                let new_x = (px + dx * damping).clamp(0.0, (self.width - 1) as f32);
                let new_y = (py + dy * damping).clamp(0.0, (self.height - 1) as f32);
                
                self.control_points[curr_base] = new_x;
                self.control_points[curr_base + 1] = new_y;
            }
        }
    }

    pub fn get_potential_field(&self) -> &[f32] {
        &self.potential_field
    }

    pub fn get_control_points(&self) -> &[f32] {
        &self.control_points
    }

    pub fn get_control_point_metadata(&self) -> (Vec<ControlPointMeta>, Vec<EdgeNodes>, Vec<u32>) {
        let total_points = self.control_points.len() / 2;
        let mut metas = vec![
            ControlPointMeta {
                prev_idx: -1,
                next_idx: -1,
                is_static: 1,
                padding: 0,
            };
            total_points
        ];
        let mut edge_nodes = vec![
            EdgeNodes {
                source: 0,
                target_node: 0,
            };
            total_points
        ];
        let mut indices = Vec::new();

        for (e_idx, &offset) in self.control_point_offsets.iter().enumerate() {
            let count = self.control_point_counts[e_idx];
            let edge = &self.edges[e_idx];

            for i in 0..count {
                let idx = offset + i;
                edge_nodes[idx] = EdgeNodes {
                    source: edge.source as u32,
                    target_node: edge.target as u32,
                };

                if i == 0 || i == count - 1 {
                    metas[idx] = ControlPointMeta {
                        prev_idx: -1,
                        next_idx: -1,
                        is_static: 1,
                        padding: 0,
                    };
                } else {
                    metas[idx] = ControlPointMeta {
                        prev_idx: (idx - 1) as i32,
                        next_idx: (idx + 1) as i32,
                        is_static: 0,
                        padding: 0,
                    };
                }

                if i < count - 1 {
                    indices.push(idx as u32);
                    indices.push((idx + 1) as u32);
                }
            }
        }

        (metas, edge_nodes, indices)
    }
}

pub fn splat_masses(width: usize, height: usize, nodes: &[Node]) -> Vec<f32> {
    let mut grid = vec![0.0; width * height];
    
    for node in nodes {
        let px = node.x.clamp(0.0, (width - 1) as f32);
        let py = node.y.clamp(0.0, (height - 1) as f32);
        
        let col = px.floor() as usize;
        let row = py.floor() as usize;
        let next_col = (col + 1).min(width - 1);
        let next_row = (row + 1).min(height - 1);
        
        let dx = px - col as f32;
        let dy = py - row as f32;
        
        let w00 = (1.0 - dx) * (1.0 - dy) * node.mass;
        let w10 = dx * (1.0 - dy) * node.mass;
        let w01 = (1.0 - dx) * dy * node.mass;
        let w11 = dx * dy * node.mass;
        
        grid[row * width + col] += w00;
        grid[row * width + next_col] += w10;
        grid[next_row * width + col] += w01;
        grid[next_row * width + next_col] += w11;
    }
    
    grid
}

pub fn generate_kernel_padded(
    width: usize,
    height: usize,
    gravity_param: f32,
    potential_max: f32,
) -> Vec<f32> {
    let pw = 2 * width;
    let ph = 2 * height;
    let mut kernel = vec![0.0; pw * ph];
    
    for y in 0..ph {
        for x in 0..pw {
            let dx = if x < width { x as f32 } else { (x as f32) - (pw as f32) };
            let dy = if y < height { y as f32 } else { (y as f32) - (ph as f32) };
            
            let d = (dx * dx + dy * dy).sqrt();
            let term = if d <= 0.0 {
                potential_max
            } else {
                (gravity_param / d).min(potential_max)
            };
            kernel[y * pw + x] = -term;
        }
    }
    
    kernel
}

pub fn convolve_fft(
    width: usize,
    height: usize,
    mass_grid: &[f32],
    kernel_padded: &[f32],
) -> Vec<f32> {
    let pw = 2 * width;
    let ph = 2 * height;
    
    let mut m_complex = vec![Complex::new(0.0, 0.0); pw * ph];
    let mut k_complex = vec![Complex::new(0.0, 0.0); pw * ph];
    
    for y in 0..height {
        for x in 0..width {
            m_complex[y * pw + x] = Complex::new(mass_grid[y * width + x], 0.0);
        }
    }
    
    for i in 0..(pw * ph) {
        k_complex[i] = Complex::new(kernel_padded[i], 0.0);
    }
    
    fft2d(&mut m_complex, pw, ph);
    fft2d(&mut k_complex, pw, ph);
    
    let mut phi_complex = vec![Complex::new(0.0, 0.0); pw * ph];
    for i in 0..(pw * ph) {
        phi_complex[i] = m_complex[i] * k_complex[i];
    }
    
    ifft2d(&mut phi_complex, pw, ph);
    
    let mut potential_field = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            potential_field[y * width + x] = phi_complex[y * pw + x].re;
        }
    }
    
    potential_field
}

pub fn calculate_forces(
    width: usize,
    height: usize,
    potential_field: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut force_field_x = vec![0.0; width * height];
    let mut force_field_y = vec![0.0; width * height];
    
    for y in 0..height {
        for x in 0..width {
            let left_x = if x > 0 { x - 1 } else { x };
            let right_x = if x + 1 < width { x + 1 } else { x };
            let dx_div = if x > 0 && x + 1 < width { 2.0 } else { 1.0 };
            
            let val_left = potential_field[y * width + left_x];
            let val_right = potential_field[y * width + right_x];
            // F = grad(Phi) (repulsive force from nodes)
            force_field_x[y * width + x] = (val_right - val_left) / dx_div;
            
            let top_y = if y > 0 { y - 1 } else { y };
            let bottom_y = if y + 1 < height { y + 1 } else { y };
            let dy_div = if y > 0 && y + 1 < height { 2.0 } else { 1.0 };
            
            let val_top = potential_field[top_y * width + x];
            let val_bottom = potential_field[bottom_y * width + x];
            // F = grad(Phi) (repulsive force from nodes)
            force_field_y[y * width + x] = (val_bottom - val_top) / dy_div;
        }
    }
    
    (force_field_x, force_field_y)
}

pub fn bilinear_interpolate_force(
    width: usize,
    height: usize,
    force_x: &[f32],
    force_y: &[f32],
    px: f32,
    py: f32,
) -> (f32, f32) {
    let px = px.clamp(0.0, (width - 1) as f32);
    let py = py.clamp(0.0, (height - 1) as f32);
    
    let col = px.floor() as usize;
    let row = py.floor() as usize;
    let next_col = (col + 1).min(width - 1);
    let next_row = (row + 1).min(height - 1);
    
    let dx = px - col as f32;
    let dy = py - row as f32;
    
    let f00_x = force_x[row * width + col];
    let f10_x = force_x[row * width + next_col];
    let f01_x = force_x[next_row * width + col];
    let f11_x = force_x[next_row * width + next_col];
    
    let fx = (1.0 - dx) * (1.0 - dy) * f00_x
        + dx * (1.0 - dy) * f10_x
        + (1.0 - dx) * dy * f01_x
        + dx * dy * f11_x;
        
    let f00_y = force_y[row * width + col];
    let f10_y = force_y[row * width + next_col];
    let f01_y = force_y[next_row * width + col];
    let f11_y = force_y[next_row * width + next_col];
    
    let fy = (1.0 - dx) * (1.0 - dy) * f00_y
        + dx * (1.0 - dy) * f10_y
        + (1.0 - dx) * dy * f01_y
        + dx * dy * f11_y;
        
    (fx, fy)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_splatting() {
        let nodes = vec![
            Node { x: 1.5, y: 1.5, mass: 10.0, degree: 1.0 }
        ];
        let grid = splat_masses(4, 4, &nodes);
        assert!((grid[5] - 2.5).abs() < 1e-5);
        assert!((grid[6] - 2.5).abs() < 1e-5);
        assert!((grid[9] - 2.5).abs() < 1e-5);
        assert!((grid[10] - 2.5).abs() < 1e-5);
        assert!((grid[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_fft2d_roundtrip() {
        let width = 8;
        let height = 8;
        let mut data = vec![Complex::new(0.0, 0.0); width * height];
        for y in 0..height {
            for x in 0..width {
                data[y * width + x] = Complex::new((x + y) as f32, 0.0);
            }
        }
        
        let original = data.clone();
        fft2d(&mut data, width, height);
        ifft2d(&mut data, width, height);
        
        for i in 0..(width * height) {
            assert!((data[i].re - original[i].re).abs() < 1e-3);
            assert!(data[i].im.abs() < 1e-3);
        }
    }

    #[test]
    fn test_gradient_forces() {
        let width = 4;
        let height = 4;
        let mut phi = vec![0.0; width * height];
        for y in 0..height {
            for x in 0..width {
                phi[y * width + x] = x as f32;
            }
        }
        
        let (fx, _fy) = calculate_forces(width, height, &phi);
        // Under F = grad(Phi), the force in the x direction is +1.0
        assert!((fx[1 * width + 1] - 1.0).abs() < 1e-5);
        assert!((fx[1 * width + 2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gravity_simulation_full() {
        let nodes = vec![
            Node { x: 2.0, y: 2.0, mass: 100.0, degree: 2.0 },
            Node { x: 6.0, y: 6.0, mass: 100.0, degree: 2.0 },
        ];
        let edges = vec![
            Edge { source: 0, target: 1 }
        ];
        
        // Spacing 1.5 on dist = sqrt(32) ~ 5.657 => floor(5.657 / 1.5) + 2 = 5 control points (10 coordinates)
        let mut sim = GravitySimulation::new(8, 8, nodes, edges, 1.5);
        assert_eq!(sim.control_points.len(), 10);
        assert_eq!(sim.control_point_offsets.len(), 1);
        assert_eq!(sim.control_point_counts.len(), 1);
        assert_eq!(sim.control_point_offsets[0], 0);
        assert_eq!(sim.control_point_counts[0], 5);
        
        sim.update_physics_fields(0.1, 16.0, 0.0);
        sim.step(0.1, 0.5, 0.9);
        
        // Control points should still be within grid bounds
        for i in 0..10 {
            assert!(sim.control_points[i] >= 0.0 && sim.control_points[i] <= 7.0);
        }
    }

    #[test]
    fn test_get_control_point_metadata() {
        let nodes = vec![
            Node { x: 0.0, y: 0.0, mass: 1.0, degree: 1.0 },
            Node { x: 10.0, y: 0.0, mass: 1.0, degree: 1.0 },
        ];
        let edges = vec![
            Edge { source: 0, target: 1 }
        ];
        
        let sim = GravitySimulation::new(12, 12, nodes, edges, 4.0);
        assert_eq!(sim.control_points.len(), 8);
        
        let (metas, edge_nodes, indices) = sim.get_control_point_metadata();
        
        assert_eq!(metas.len(), 4);
        assert_eq!(edge_nodes.len(), 4);
        assert_eq!(indices.len(), 6);
        assert_eq!(indices, vec![0, 1, 1, 2, 2, 3]);
        
        // Check static ends
        assert_eq!(metas[0].is_static, 1);
        assert_eq!(metas[0].prev_idx, -1);
        assert_eq!(metas[0].next_idx, -1);
        
        assert_eq!(metas[3].is_static, 1);
        assert_eq!(metas[3].prev_idx, -1);
        assert_eq!(metas[3].next_idx, -1);
        
        // Check dynamic inner points
        assert_eq!(metas[1].is_static, 0);
        assert_eq!(metas[1].prev_idx, 0);
        assert_eq!(metas[1].next_idx, 2);
        
        assert_eq!(metas[2].is_static, 0);
        assert_eq!(metas[2].prev_idx, 1);
        assert_eq!(metas[2].next_idx, 3);
        
        // Check edge nodes info
        for i in 0..4 {
            assert_eq!(edge_nodes[i].source, 0);
            assert_eq!(edge_nodes[i].target_node, 1);
        }
    }

    #[test]
    fn test_pseudo_newtonian_direct_sum() {
        let nodes = vec![
            Node { x: 2.0, y: 2.0, mass: 10.0, degree: 1.0 },
        ];
        let edges = vec![
            Edge { source: 0, target: 0 }
        ];
        let mut sim = GravitySimulation::new(4, 4, nodes, edges, 1.5);
        
        // gravity_param = 1.0, softening = 1.0, alpha = 0.5
        // scale = 4.0 / 256.0 = 0.015625
        // softening_scaled = 1.0 * 0.015625 = 0.015625
        // At (2, 2), distance d = 0.0.
        // d - alpha * mass = 0.0 - 0.5 * 10.0 = -5.0.
        // denom = max(-5.0, softening_scaled) = 0.015625.
        // pot_sum = -1.0 * 10.0 / 0.015625 = -640.0.
        // potential at (2, 2) = pot_sum * scale = -640.0 * 0.015625 = -10.0.
        sim.update_physics_fields(1.0, 10.0, 0.5);
        
        let pot = sim.get_potential_field();
        let idx = 2 * 4 + 2; // (2, 2)
        assert!((pot[idx] - (-10.0)).abs() < 1e-5);
    }
}
