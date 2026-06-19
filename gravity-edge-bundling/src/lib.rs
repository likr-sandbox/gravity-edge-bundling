use wasm_bindgen::prelude::*;
use crate::simulation::{GravitySimulation, Node, Edge};

pub mod fft2d;
pub mod simulation;

#[wasm_bindgen]
pub struct SimulationState {
    inner: GravitySimulation,
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
        Ok(SimulationState { inner })
    }
    
    pub fn update_nodes(&mut self, nodes_js: JsValue) -> Result<(), JsValue> {
        let nodes: Vec<Node> = serde_wasm_bindgen::from_value(nodes_js)?;
        self.inner.update_nodes(nodes);
        Ok(())
    }
    
    pub fn reset_control_points(&mut self) {
        self.inner.reset_control_points();
    }
    
    pub fn update_physics_fields(&mut self, g_constant: f32, softening_epsilon: f32, range_scale: f32) {
        self.inner.update_physics_fields(g_constant, softening_epsilon, range_scale);
    }
    
    pub fn step(&mut self, spring_k: f32, dt: f32, damping: f32) {
        self.inner.step(spring_k, dt, damping);
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
