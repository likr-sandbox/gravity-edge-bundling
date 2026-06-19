use wasm_bindgen::prelude::*;
use crate::simulation::{GravitySimulation, Node, Edge};

use crate::webgpu::{SimParams, NodeVertex, WgpuContext};

pub mod fft2d;
pub mod simulation;
pub mod webgpu;

#[wasm_bindgen]
pub struct SimulationState {
    inner: GravitySimulation,
    wgpu_ctx: Option<WgpuContext>,
    
    // cached physical parameters
    spring_k: f32,
    dt: f32,
    damping: f32,
    
    // cached render parameters
    edge_opacity: f32,
    heatmap_opacity: f32,
    hovered_node: Option<u32>,
    show_heatmap: bool,
    show_nodes: bool,
    show_bundled_edges: bool,
}

#[wasm_bindgen]
impl SimulationState {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        nodes_js: JsValue,
        edges_js: JsValue,
        control_point_spacing: f32,
    ) -> Result<SimulationState, JsValue> {
        console_error_panic_hook::set_once();
        
        let nodes: Vec<Node> = serde_wasm_bindgen::from_value(nodes_js)?;
        let edges: Vec<Edge> = serde_wasm_bindgen::from_value(edges_js)?;
        
        let inner = GravitySimulation::new(width, height, nodes, edges, control_point_spacing);
        Ok(SimulationState {
            inner,
            wgpu_ctx: None,
            spring_k: 0.05,
            dt: 0.5,
            damping: 0.95,
            edge_opacity: 0.1,
            heatmap_opacity: 0.85,
            hovered_node: None,
            show_heatmap: true,
            show_nodes: true,
            show_bundled_edges: true,
        })
    }


    
    // Standalone create_simulation_state handles async WebGPU instantiation
    
    fn sync_gpu_buffers(&mut self) {
        if let Some(ctx) = &mut self.wgpu_ctx {
            let (metas, edge_nodes, indices) = self.inner.get_control_point_metadata();
            ctx.update_buffers(
                &self.inner.control_points,
                &metas,
                &edge_nodes,
                &indices,
            );
        }
    }
    
    fn sync_nodes_buffer(&mut self) {
        if let Some(ctx) = &mut self.wgpu_ctx {
            let node_vertices: Vec<NodeVertex> = self.inner.nodes.iter().enumerate().map(|(idx, n)| {
                let r = (n.degree + 1.0).log2() * 1.3;
                let r_clamped = r.max(2.0);
                let is_hovered = if Some(idx as u32) == self.hovered_node { 1 } else { 0 };
                NodeVertex {
                    pos: [n.x, n.y],
                    r: r_clamped,
                    is_hovered,
                }
            }).collect();
            ctx.update_nodes(&node_vertices);
        }
    }
    
    pub fn update_nodes(&mut self, nodes_js: JsValue) -> Result<(), JsValue> {
        let nodes: Vec<Node> = serde_wasm_bindgen::from_value(nodes_js)?;
        self.inner.update_nodes(nodes);
        self.sync_nodes_buffer();
        Ok(())
    }
    
    pub fn reset_control_points(&mut self) {
        self.inner.reset_control_points();
        self.sync_gpu_buffers();
    }
    
    pub fn update_physics_fields(&mut self, g_constant: f32, softening_epsilon: f32, range_scale: f32) {
        self.inner.update_physics_fields(g_constant, softening_epsilon, range_scale);
        
        if let Some(ctx) = &mut self.wgpu_ctx {
            ctx.update_force_field(&self.inner.force_field_x, &self.inner.force_field_y);
            ctx.update_potential_heatmap(&self.inner.potential_field);
        }
    }
    
    pub fn step(&mut self, spring_k: f32, dt: f32, damping: f32) {
        self.spring_k = spring_k;
        self.dt = dt;
        self.damping = damping;
        
        if let Some(ctx) = &mut self.wgpu_ctx {
            let mut min_pot = f32::INFINITY;
            let mut max_pot = f32::NEG_INFINITY;
            for &v in &self.inner.potential_field {
                if v < min_pot { min_pot = v; }
                if v > max_pot { max_pot = v; }
            }
            
            let canvas_size = ctx.canvas_size();
            let params = SimParams {
                spring_k,
                dt,
                damping,
                grid_width: self.inner.width as f32,
                grid_height: self.inner.height as f32,
                edge_opacity: self.edge_opacity,
                heatmap_opacity: self.heatmap_opacity,
                hovered_node: self.hovered_node.map(|x| x as i32).unwrap_or(-1),
                show_heatmap: if self.show_heatmap { 1 } else { 0 },
                show_nodes: if self.show_nodes { 1 } else { 0 },
                show_bundled_edges: if self.show_bundled_edges { 1 } else { 0 },
                min_potential: min_pot,
                max_potential: max_pot,
                canvas_width: canvas_size.0 as f32,
                canvas_height: canvas_size.1 as f32,
                padding: 0,
            };
            ctx.update_params(&params);
            ctx.step();
            return;
        }
        
        self.inner.step(spring_k, dt, damping);
    }
    
    pub fn render(
        &mut self,
        hovered_node: Option<u32>,
        edge_opacity: f32,
        heatmap_opacity: f32,
        show_heatmap: bool,
        show_nodes: bool,
        show_bundled_edges: bool,
    ) {
        self.edge_opacity = edge_opacity;
        self.heatmap_opacity = heatmap_opacity;
        
        let hovered_changed = self.hovered_node != hovered_node;
        self.hovered_node = hovered_node;
        self.show_heatmap = show_heatmap;
        self.show_nodes = show_nodes;
        self.show_bundled_edges = show_bundled_edges;
        
        if hovered_changed {
            self.sync_nodes_buffer();
        }
        
        if let Some(ctx) = &mut self.wgpu_ctx {
            let mut min_pot = f32::INFINITY;
            let mut max_pot = f32::NEG_INFINITY;
            for &v in &self.inner.potential_field {
                if v < min_pot { min_pot = v; }
                if v > max_pot { max_pot = v; }
            }
            
            let canvas_size = ctx.canvas_size();
            let params = SimParams {
                spring_k: self.spring_k,
                dt: self.dt,
                damping: self.damping,
                grid_width: self.inner.width as f32,
                grid_height: self.inner.height as f32,
                edge_opacity: self.edge_opacity,
                heatmap_opacity: self.heatmap_opacity,
                hovered_node: self.hovered_node.map(|x| x as i32).unwrap_or(-1),
                show_heatmap: if self.show_heatmap { 1 } else { 0 },
                show_nodes: if self.show_nodes { 1 } else { 0 },
                show_bundled_edges: if self.show_bundled_edges { 1 } else { 0 },
                min_potential: min_pot,
                max_potential: max_pot,
                canvas_width: canvas_size.0 as f32,
                canvas_height: canvas_size.1 as f32,
                padding: 0,
            };
            ctx.update_params(&params);
            ctx.render();
        }
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        if let Some(ctx) = &mut self.wgpu_ctx {
            ctx.resize(width, height);
        }
    }
    
    pub fn get_potential_field(&self) -> Vec<f32> {
        self.inner.get_potential_field().to_vec()
    }
    
    pub fn get_control_points(&self) -> Vec<f32> {
        self.inner.get_control_points().to_vec()
    }

    pub fn get_control_point_offsets(&self) -> Vec<u32> {
        self.inner.control_point_offsets.iter().map(|&x| x as u32).collect()
    }

    pub fn get_control_point_counts(&self) -> Vec<u32> {
        self.inner.control_point_counts.iter().map(|&x| x as u32).collect()
    }

}

impl SimulationState {
    pub fn new_native(
        width: usize,
        height: usize,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        control_point_spacing: f32,
    ) -> SimulationState {
        let inner = GravitySimulation::new(width, height, nodes, edges, control_point_spacing);
        SimulationState {
            inner,
            wgpu_ctx: None,
            spring_k: 0.05,
            dt: 0.5,
            damping: 0.95,
            edge_opacity: 0.1,
            heatmap_opacity: 0.85,
            hovered_node: None,
            show_heatmap: true,
            show_nodes: true,
            show_bundled_edges: true,
        }
    }

    pub async fn init_wgpu_headless(&mut self) -> Result<(), String> {
        let wgpu_ctx = WgpuContext::new_headless(self.inner.width as u32, self.inner.height as u32).await?;
        self.wgpu_ctx = Some(wgpu_ctx);
        self.sync_gpu_buffers();
        self.sync_nodes_buffer();
        Ok(())
    }
    
    pub async fn get_control_points_gpu(&self) -> Result<Vec<f32>, String> {
        if let Some(ctx) = &self.wgpu_ctx {
            ctx.read_positions().await
        } else {
            Err("WebGPU context not initialized".to_string())
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn create_simulation_state(
    canvas: web_sys::HtmlCanvasElement,
    width: usize,
    height: usize,
    nodes_js: JsValue,
    edges_js: JsValue,
    control_point_spacing: f32,
) -> Result<SimulationState, JsValue> {
    console_error_panic_hook::set_once();
    
    let nodes: Vec<Node> = serde_wasm_bindgen::from_value(nodes_js)?;
    let edges: Vec<Edge> = serde_wasm_bindgen::from_value(edges_js)?;
    
    let inner = GravitySimulation::new(width, height, nodes, edges, control_point_spacing);
    
    let wgpu_ctx = WgpuContext::new(canvas, width as u32, height as u32)
        .await
        .map_err(|e| JsValue::from_str(&e))?;
        
    let mut state = SimulationState {
        inner,
        wgpu_ctx: Some(wgpu_ctx),
        spring_k: 0.05,
        dt: 0.5,
        damping: 0.95,
        edge_opacity: 0.1,
        heatmap_opacity: 0.85,
        hovered_node: None,
        show_heatmap: true,
        show_nodes: true,
        show_bundled_edges: true,
    };
    
    state.sync_gpu_buffers();
    state.sync_nodes_buffer();
    
    Ok(state)
}

