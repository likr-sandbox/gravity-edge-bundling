use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub spring_k: f32,
    pub dt: f32,
    pub damping: f32,
    pub grid_width: f32,
    pub grid_height: f32,
    pub edge_opacity: f32,
    pub heatmap_opacity: f32,
    pub hovered_node: i32,
    pub show_heatmap: u32,
    pub show_nodes: u32,
    pub show_bundled_edges: u32,
    pub potential_max: f32,
    pub canvas_width: f32,
    pub canvas_height: f32,
    pub gravity_param: f32,
    pub gravity_alpha: f32,
    pub padding1: u32,
    pub padding2: u32,
    pub padding3: u32,
    pub padding4: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeSim {
    pub pos: [f32; 2],
    pub mass: f32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ControlPointMeta {
    pub prev_idx: i32,
    pub next_idx: i32,
    pub is_static: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeNodes {
    pub source: u32,
    pub target_node: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeVertex {
    pub pos: [f32; 2],
    pub r: f32,
    pub is_hovered: u32,
}

pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: Option<wgpu::Surface<'static>>,
    pub config: Option<wgpu::SurfaceConfiguration>,
    
    // Pipelines
    compute_pipeline: wgpu::ComputePipeline,
    gravity_pipeline: wgpu::ComputePipeline,
    heatmap_pipeline: Option<wgpu::RenderPipeline>,
    edge_pipeline: Option<wgpu::RenderPipeline>,
    node_pipeline: Option<wgpu::RenderPipeline>,
    
    // Buffers
    positions_a: wgpu::Buffer,
    positions_b: wgpu::Buffer,
    meta_buffer: wgpu::Buffer,
    edge_nodes_buffer: wgpu::Buffer,
    indices_buffer: wgpu::Buffer,
    nodes_buffer: wgpu::Buffer,
    nodes_sim_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    
    // Textures & Samplers
    heatmap_texture: wgpu::Texture,
    heatmap_view: wgpu::TextureView,
    force_texture: wgpu::Texture,
    force_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    
    // Bind Groups
    compute_bind_group_a: wgpu::BindGroup,
    compute_bind_group_b: wgpu::BindGroup,
    gravity_bind_group: Option<wgpu::BindGroup>,
    render_bind_group: wgpu::BindGroup,
    render_params_bind_group: wgpu::BindGroup,
    
    gravity_bind_group_layout: wgpu::BindGroupLayout,
    render_resources_layout: wgpu::BindGroupLayout,
    
    // State
    is_a_source: bool,
    num_control_points: u32,
    num_indices: u32,
    num_nodes: u32,
    
    grid_width: u32,
    grid_height: u32,
}

const COMPUTE_SHADER: &str = r#"
struct Params {
    spring_k: f32,
    dt: f32,
    damping: f32,
    grid_width: f32,
    grid_height: f32,
    edge_opacity: f32,
    heatmap_opacity: f32,
    hovered_node: i32,
    show_heatmap: u32,
    show_nodes: u32,
    show_bundled_edges: u32,
    potential_max: f32,
    canvas_width: f32,
    canvas_height: f32,
    gravity_param: f32,
    gravity_alpha: f32,
}

struct Meta {
    prev_idx: i32,
    next_idx: i32,
    is_static: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> metas: array<Meta>;
@group(0) @binding(2) var<storage, read> prev_positions: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(4) var force_texture: texture_2d<f32>;
@group(0) @binding(5) var force_sampler: sampler;

fn sample_force_bilinear(coords: vec2<f32>) -> vec2<f32> {
    let size = vec2<f32>(params.grid_width, params.grid_height);
    let pixel = coords * size - 0.5;
    let grid_x = floor(pixel.x);
    let grid_y = floor(pixel.y);
    let f = fract(pixel);
    
    let x0 = clamp(i32(grid_x), 0, i32(params.grid_width) - 1);
    let x1 = clamp(i32(grid_x) + 1, 0, i32(params.grid_width) - 1);
    let y0 = clamp(i32(grid_y), 0, i32(params.grid_height) - 1);
    let y1 = clamp(i32(grid_y) + 1, 0, i32(params.grid_height) - 1);
    
    let t00 = textureLoad(force_texture, vec2<i32>(x0, y0), 0).xy;
    let t10 = textureLoad(force_texture, vec2<i32>(x1, y0), 0).xy;
    let t01 = textureLoad(force_texture, vec2<i32>(x0, y1), 0).xy;
    let t11 = textureLoad(force_texture, vec2<i32>(x1, y1), 0).xy;
    
    let top = mix(t00, t10, f.x);
    let bottom = mix(t01, t11, f.x);
    return mix(top, bottom, f.y);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&positions)) {
        return;
    }
    
    let m_data = metas[idx];
    let curr_pos = prev_positions[idx];
    
    if (m_data.is_static != 0u) {
        positions[idx] = curr_pos;
        return;
    }
    
    let prev_pos = prev_positions[m_data.prev_idx];
    let next_pos = prev_positions[m_data.next_idx];
    let f_spring = params.spring_k * (prev_pos + next_pos - 2.0 * curr_pos);
    
    let tex_coord = curr_pos / vec2<f32>(params.grid_width, params.grid_height);
    let f_grav = sample_force_bilinear(tex_coord);
    
    let f_total = f_spring + f_grav;
    var disp = f_total * params.dt;
    
    let max_disp = 5.0;
    let disp_len = length(disp);
    if (disp_len > max_disp) {
        disp = (disp / disp_len) * max_disp;
    }
    
    var new_pos = curr_pos + disp * params.damping;
    new_pos = clamp(new_pos, vec2<f32>(0.0, 0.0), vec2<f32>(params.grid_width - 1.0, params.grid_height - 1.0));
    
    positions[idx] = new_pos;
}
"#;

const RENDER_SHADER: &str = r#"
struct Params {
    spring_k: f32,
    dt: f32,
    damping: f32,
    grid_width: f32,
    grid_height: f32,
    edge_opacity: f32,
    heatmap_opacity: f32,
    hovered_node: i32,
    show_heatmap: u32,
    show_nodes: u32,
    show_bundled_edges: u32,
    potential_max: f32,
    canvas_width: f32,
    canvas_height: f32,
    gravity_param: f32,
    gravity_alpha: f32,
}

struct EdgeNodes {
    source: u32,
    target_node: u32,
}

struct NodeVertex {
    pos: vec2<f32>,
    r: f32,
    is_hovered: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

// Heatmap resources
@group(1) @binding(0) var heatmap_texture: texture_2d<f32>;
@group(1) @binding(1) var heatmap_sampler: sampler;

// Edge resources
@group(1) @binding(2) var<storage, read> positions: array<vec2<f32>>;
@group(1) @binding(3) var<storage, read> edge_nodes: array<EdgeNodes>;
@group(1) @binding(4) var<storage, read> indices: array<u32>;

// Node resources
@group(1) @binding(5) var<storage, read> nodes: array<NodeVertex>;


// --- Heatmap Render ---

struct QuadOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_heatmap(@builtin(vertex_index) vertex_idx: u32) -> QuadOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0)
    );
    
    var out: QuadOutput;
    out.position = vec4<f32>(pos[vertex_idx], 0.0, 1.0);
    out.uv = uv[vertex_idx];
    return out;
}

fn sample_heatmap_bilinear(coords: vec2<f32>) -> f32 {
    let size = vec2<f32>(params.grid_width, params.grid_height);
    let pixel = coords * size - 0.5;
    let grid_x = floor(pixel.x);
    let grid_y = floor(pixel.y);
    let f = fract(pixel);
    
    let x0 = clamp(i32(grid_x), 0, i32(params.grid_width) - 1);
    let x1 = clamp(i32(grid_x) + 1, 0, i32(params.grid_width) - 1);
    let y0 = clamp(i32(grid_y), 0, i32(params.grid_height) - 1);
    let y1 = clamp(i32(grid_y) + 1, 0, i32(params.grid_height) - 1);
    
    let t00 = textureLoad(heatmap_texture, vec2<i32>(x0, y0), 0).r;
    let t10 = textureLoad(heatmap_texture, vec2<i32>(x1, y0), 0).r;
    let t01 = textureLoad(heatmap_texture, vec2<i32>(x0, y1), 0).r;
    let t11 = textureLoad(heatmap_texture, vec2<i32>(x1, y1), 0).r;
    
    let top = mix(t00, t10, f.x);
    let bottom = mix(t01, t11, f.x);
    return mix(top, bottom, f.y);
}

@fragment
fn fs_heatmap(in: QuadOutput) -> @location(0) vec4<f32> {
    if (params.show_heatmap == 0u) {
        discard;
    }
    
    let val = sample_heatmap_bilinear(in.uv);
    var t = 0.0;
    if (params.potential_max > 0.0) {
        t = clamp((-val) / params.potential_max, 0.0, 1.0);
    }
    
    if (t <= 0.02) {
        discard;
    }
    
    let r = t * t * 140.0 / 255.0;
    var g = t * 180.0;
    if (t > 0.6) {
        g = g + (t - 0.6) * 150.0;
    }
    g = g / 255.0;
    let b = (80.0 + t * 175.0) / 255.0;
    let a = t * 230.0 / 255.0 * params.heatmap_opacity;
    
    return vec4<f32>(r, g, b, a);
}


// --- Edge Render ---

struct EdgeOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) is_highlighted: u32,
}

@vertex
fn vs_edge(@builtin(vertex_index) vertex_idx: u32) -> EdgeOutput {
    let pt_idx = indices[vertex_idx];
    let pos = positions[pt_idx];
    
    let x_ndc = (pos.x / params.grid_width) * 2.0 - 1.0;
    let y_ndc = 1.0 - (pos.y / params.grid_height) * 2.0;
    
    let en = edge_nodes[pt_idx];
    var is_high = 0u;
    if (params.hovered_node >= 0 && 
        (i32(en.source) == params.hovered_node || i32(en.target_node) == params.hovered_node)) {
        is_high = 1u;
    }
    
    var out: EdgeOutput;
    out.position = vec4<f32>(x_ndc, y_ndc, 0.0, 1.0);
    out.is_highlighted = is_high;
    return out;
}

@fragment
fn fs_edge(in: EdgeOutput) -> @location(0) vec4<f32> {
    if (params.show_bundled_edges == 0u) {
        discard;
    }
    
    if (in.is_highlighted != 0u) {
        return vec4<f32>(0.0, 1.0, 0.784, 0.8);
    } else {
        return vec4<f32>(0.0, 0.824, 1.0, params.edge_opacity);
    }
}


// --- Node Render ---

struct NodeOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) @interpolate(flat) is_hovered: u32,
}

@vertex
fn vs_node(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> NodeOutput {
    var quad_pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    let l_pos = quad_pos[vertex_idx % 4u];
    
    let node = nodes[instance_idx];
    
    let center_x_ndc = (node.pos.x / params.grid_width) * 2.0 - 1.0;
    let center_y_ndc = 1.0 - (node.pos.y / params.grid_height) * 2.0;
    
    let radius = select(node.r, node.r + 3.0, node.is_hovered != 0u);
    let offset_x = l_pos.x * radius * (2.0 / params.canvas_width);
    let offset_y = l_pos.y * radius * (2.0 / params.canvas_height);
    
    var out: NodeOutput;
    out.position = vec4<f32>(center_x_ndc + offset_x, center_y_ndc + offset_y, 0.0, 1.0);
    out.local_uv = l_pos;
    out.is_hovered = node.is_hovered;
    return out;
}

@fragment
fn fs_node(in: NodeOutput) -> @location(0) vec4<f32> {
    if (params.show_nodes == 0u) {
        discard;
    }
    
    let dist = length(in.local_uv);
    if (dist > 1.0) {
        discard;
    }
    
    let width = fwidth(dist);
    let alpha = 1.0 - smoothstep(1.0 - width, 1.0, dist);
    
    if (in.is_hovered != 0u) {
        return vec4<f32>(0.0, 1.0, 0.784, alpha);
    } else {
        return vec4<f32>(1.0, 1.0, 1.0, alpha * 0.75);
    }
}
"#;

const GRAVITY_COMPUTE_SHADER: &str = r#"
struct Params {
    spring_k: f32,
    dt: f32,
    damping: f32,
    grid_width: f32,
    grid_height: f32,
    edge_opacity: f32,
    heatmap_opacity: f32,
    hovered_node: i32,
    show_heatmap: u32,
    show_nodes: u32,
    show_bundled_edges: u32,
    potential_max: f32,
    canvas_width: f32,
    canvas_height: f32,
    gravity_param: f32,
    gravity_alpha: f32,
}

struct NodeSim {
    pos: vec2<f32>,
    mass: f32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<NodeSim>;
@group(0) @binding(2) var heatmap_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var force_texture: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = u32(params.grid_width);
    let height = u32(params.grid_height);
    
    if (x >= width || y >= height) {
        return;
    }
    
    let coords = vec2<f32>(f32(x), f32(y));
    let scale = params.grid_width / 256.0;
    let potential_max_scaled = max(params.potential_max / scale, 1e-5);
    
    var potential = 0.0;
    var force = vec2<f32>(0.0, 0.0);
    
    let num_nodes = arrayLength(&nodes);
    
    for (var i = 0u; i < num_nodes; i = i + 1u) {
        let node = nodes[i];
        let diff = coords - node.pos;
        let d = length(diff);
        let softening_scaled = (params.gravity_param * node.mass) / potential_max_scaled;
        let denom = max(d - params.gravity_alpha * node.mass, softening_scaled);
        
        potential = potential - (params.gravity_param * node.mass) / denom;
        
        if (d > 0.0) {
            if (d - params.gravity_alpha * node.mass > softening_scaled) {
                let f_mag = (params.gravity_param * node.mass) / (denom * denom);
                force = force + f_mag * (diff / d);
            }
        }
    }
    
    let pot_final = potential * scale;
    let force_final = force * scale * scale;
    
    textureStore(heatmap_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(pot_final, 0.0, 0.0, 0.0));
    textureStore(force_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(force_final.x, force_final.y, 0.0, 0.0));
}
"#;

impl WgpuContext {
    #[cfg(target_arch = "wasm32")]
    pub async fn new(
        canvas: web_sys::HtmlCanvasElement,
        grid_width: u32,
        grid_height: u32,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| format!("Failed to create surface: {:?}", e))?;
            
        let canvas_width = canvas.width();
        let canvas_height = canvas.height();
        
        Self::new_internal(
            instance,
            Some(surface),
            canvas_width,
            canvas_height,
            grid_width,
            grid_height,
        )
        .await
    }

    pub async fn new_headless(
        grid_width: u32,
        grid_height: u32,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        Self::new_internal(
            instance,
            None,
            800,
            600,
            grid_width,
            grid_height,
        )
        .await
    }

    async fn new_internal(
        instance: wgpu::Instance,
        surface: Option<wgpu::Surface<'static>>,
        canvas_width: u32,
        canvas_height: u32,
        grid_width: u32,
        grid_height: u32,
    ) -> Result<Self, String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "Failed to request adapter".to_string())?;
            
        let mut required_features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
            required_features |= wgpu::Features::FLOAT32_FILTERABLE;
        }
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features,
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to request device: {:?}", e))?;
            
        let config = if let Some(ref surface) = surface {
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: wgpu::TextureFormat::Rgba8Unorm,
                width: canvas_width,
                height: canvas_height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&device, &config);
            Some(config)
        } else {
            None
        };
        
        // --- Shader Modules ---
        let compute_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
        });
        
        let render_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
        });
        
        // --- Buffers ---
        let positions_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions A"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let positions_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions B"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Metas"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let edge_nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Edge Nodes"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indices"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nodes"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let nodes_sim_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nodes Sim"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Uniform"),
            size: std::mem::size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Heatmap potential texture
        let heatmap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Heatmap Texture"),
            size: wgpu::Extent3d {
                width: grid_width,
                height: grid_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let heatmap_view = heatmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Force field texture
        let force_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Force Texture"),
            size: wgpu::Extent3d {
                width: grid_width,
                height: grid_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let force_view = force_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Nearest Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // --- Bind Group Layouts ---
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        
        let gravity_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gravity Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        
        let render_params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Params Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let render_resources_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Resources Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // --- Pipeline Layouts ---
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let gravity_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gravity Layout"),
            bind_group_layouts: &[&gravity_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Layout"),
            bind_group_layouts: &[&render_params_layout, &render_resources_layout],
            push_constant_ranges: &[],
        });
        
        // --- Pipelines ---
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_module,
            entry_point: "main",
        });
        
        let gravity_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gravity Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(GRAVITY_COMPUTE_SHADER.into()),
        });
        
        let gravity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gravity Pipeline"),
            layout: Some(&gravity_pipeline_layout),
            module: &gravity_module,
            entry_point: "main",
        });
        
        let (heatmap_pipeline, edge_pipeline, node_pipeline) = if let Some(ref config) = config {
            let heatmap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Heatmap Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: "vs_heatmap",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: "fs_heatmap",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
            
            let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Edge Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: "vs_edge",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: "fs_edge",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
            
            let node_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Node Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: "vs_node",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: "fs_node",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
            
            (Some(heatmap_pipeline), Some(edge_pipeline), Some(node_pipeline))
        } else {
            (None, None, None)
        };
        
        // Initial Bind Groups
        let compute_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BG A"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: meta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: positions_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: positions_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&force_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        
        let compute_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BG B"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: meta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: positions_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: positions_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&force_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Resources BG"),
            layout: &render_resources_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&heatmap_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: positions_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: edge_nodes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: nodes_buffer.as_entire_binding() },
            ],
        });
        
        let render_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Params BG"),
            layout: &render_params_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            ],
        });
        
        let gravity_bind_group = None; // Will be created when nodes are uploaded
        
        Ok(Self {
            device,
            queue,
            surface,
            config,
            compute_pipeline,
            gravity_pipeline,
            heatmap_pipeline,
            edge_pipeline,
            node_pipeline,
            positions_a,
            positions_b,
            meta_buffer,
            edge_nodes_buffer,
            indices_buffer,
            nodes_buffer,
            nodes_sim_buffer,
            params_buffer,
            heatmap_texture,
            heatmap_view,
            force_texture,
            force_view,
            sampler,
            compute_bind_group_a,
            compute_bind_group_b,
            gravity_bind_group,
            render_bind_group,
            render_params_bind_group,
            gravity_bind_group_layout,
            render_resources_layout,
            is_a_source: true,
            num_control_points: 0,
            num_indices: 0,
            num_nodes: 0,
            grid_width,
            grid_height,
        })
    }
    
    pub fn canvas_size(&self) -> (u32, u32) {
        if let Some(ref config) = self.config {
            (config.width, config.height)
        } else {
            (800, 600)
        }
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            if let Some(ref mut config) = self.config {
                config.width = width;
                config.height = height;
                if let Some(ref surface) = self.surface {
                    surface.configure(&self.device, config);
                }
            }
        }
    }
    
    pub fn update_buffers(
        &mut self,
        positions: &[f32],
        metas: &[ControlPointMeta],
        edge_nodes: &[EdgeNodes],
        indices: &[u32],
    ) {
        self.num_control_points = (positions.len() / 2) as u32;
        self.num_indices = indices.len() as u32;
        
        let pos_size = (positions.len() * 4) as u64;
        
        self.positions_a = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Positions A"),
            contents: bytemuck::cast_slice(positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        
        self.positions_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions B"),
            size: pos_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        self.meta_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Metas"),
            contents: bytemuck::cast_slice(metas),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        self.edge_nodes_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Edge Nodes"),
            contents: bytemuck::cast_slice(edge_nodes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        self.indices_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Indices"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        self.is_a_source = true;
        
        self.recreate_bind_groups();
    }
    
    pub fn update_nodes(&mut self, node_data: &[NodeVertex], node_sim_data: &[NodeSim]) {
        self.num_nodes = node_data.len() as u32;
        
        self.nodes_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nodes"),
            contents: bytemuck::cast_slice(node_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        self.nodes_sim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nodes Sim"),
            contents: bytemuck::cast_slice(node_sim_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        self.recreate_bind_groups();
    }
    
    fn recreate_bind_groups(&mut self) {
        let compute_bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        
        self.compute_bind_group_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BG A"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.meta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.positions_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.positions_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.force_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        
        self.compute_bind_group_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BG B"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.meta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.positions_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.positions_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.force_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        
        let active_positions = if self.is_a_source {
            &self.positions_a
        } else {
            &self.positions_b
        };
        
        self.render_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Resources BG"),
            layout: &self.render_resources_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.heatmap_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: active_positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.edge_nodes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.nodes_buffer.as_entire_binding() },
            ],
        });
        
        self.gravity_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gravity BG"),
            layout: &self.gravity_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.nodes_sim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.heatmap_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.force_view) },
            ],
        }));
    }
    
    pub fn update_physics_fields_gpu(&mut self) {
        if self.num_nodes == 0 {
            return;
        }
        
        let gravity_bind_group = match &self.gravity_bind_group {
            Some(bg) => bg,
            None => return,
        };
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gravity Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gravity Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.gravity_pipeline);
            compute_pass.set_bind_group(0, gravity_bind_group, &[]);
            
            let workgroup_x = (self.grid_width + 15) / 16;
            let workgroup_y = (self.grid_height + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
    }
    
    pub fn update_params(&mut self, params: &SimParams) {
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }
    
    pub fn update_force_field(&mut self, force_x: &[f32], force_y: &[f32]) {
        let mut force_data = Vec::with_capacity(force_x.len() * 2);
        for i in 0..force_x.len() {
            force_data.push(force_x[i]);
            force_data.push(force_y[i]);
        }
        
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.force_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&force_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.grid_width * 8),
                rows_per_image: Some(self.grid_height),
            },
            wgpu::Extent3d {
                width: self.grid_width,
                height: self.grid_height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    pub fn update_potential_heatmap(&mut self, potential: &[f32]) {
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.heatmap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(potential),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.grid_width * 4),
                rows_per_image: Some(self.grid_height),
            },
            wgpu::Extent3d {
                width: self.grid_width,
                height: self.grid_height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    pub fn step(&mut self) {
        if self.num_control_points == 0 {
            return;
        }
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Step Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Step Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            if self.is_a_source {
                compute_pass.set_bind_group(0, &self.compute_bind_group_a, &[]);
            } else {
                compute_pass.set_bind_group(0, &self.compute_bind_group_b, &[]);
            }
            
            let workgroup_count = (self.num_control_points + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        self.is_a_source = !self.is_a_source;
        self.recreate_bind_groups();
    }
    
    pub fn render(&mut self) {
        let surface = match &self.surface {
            Some(s) => s,
            None => return,
        };
        
        let frame = match surface.get_current_texture() {
            Ok(texture) => texture,
            Err(e) => {
                let msg = format!("Failed to acquire next swap chain texture: {:?}", e);
                #[cfg(target_arch = "wasm32")]
                web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(&msg));
                #[cfg(not(target_arch = "wasm32"))]
                eprintln!("{}", msg);
                return;
            }
        };
        
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 6.0 / 255.0,
                            g: 9.0 / 255.0,
                            b: 17.0 / 255.0,
                            a: 1.0,
                        }), // #060911
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            if let Some(ref heatmap_pipeline) = self.heatmap_pipeline {
                render_pass.set_pipeline(heatmap_pipeline);
                render_pass.set_bind_group(0, &self.render_params_bind_group, &[]);
                render_pass.set_bind_group(1, &self.render_bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
            
            if self.num_indices > 0 {
                if let Some(ref edge_pipeline) = self.edge_pipeline {
                    render_pass.set_pipeline(edge_pipeline);
                    render_pass.set_bind_group(0, &self.render_params_bind_group, &[]);
                    render_pass.set_bind_group(1, &self.render_bind_group, &[]);
                    render_pass.draw(0..self.num_indices, 0..1);
                }
            }
            
            if self.num_nodes > 0 {
                if let Some(ref node_pipeline) = self.node_pipeline {
                    render_pass.set_pipeline(node_pipeline);
                    render_pass.set_bind_group(0, &self.render_params_bind_group, &[]);
                    render_pass.set_bind_group(1, &self.render_bind_group, &[]);
                    render_pass.draw(0..4, 0..self.num_nodes);
                }
            }
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }

    pub async fn read_positions(&self) -> Result<Vec<f32>, String> {
        if self.num_control_points == 0 {
            return Ok(Vec::new());
        }
        
        let size = (self.num_control_points * 2 * 4) as u64; // index * x,y * f32
        
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Positions Encoder"),
        });
        
        let active_positions = if self.is_a_source {
            &self.positions_a
        } else {
            &self.positions_b
        };
        
        encoder.copy_buffer_to_buffer(active_positions, 0, &staging_buffer, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));
        
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        
        rx.recv()
            .map_err(|e| format!("Channel receive error: {:?}", e))?
            .map_err(|e| format!("Buffer map error: {:?}", e))?;
            
        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
}
