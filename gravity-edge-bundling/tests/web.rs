use wasm_bindgen_test::*;
use gravity_edge_bundling::SimulationState;


#[wasm_bindgen_test]
fn test_state_construction_and_fields() {
    let width = 8;
    let height = 8;
    
    // JSON arrays representation of Nodes and Edges
    let nodes_json = r#"[
        {"x": 2.0, "y": 2.0, "mass": 10.0, "degree": 1.0},
        {"x": 6.0, "y": 6.0, "mass": 20.0, "degree": 2.0}
    ]"#;
    let edges_json = r#"[
        {"source": 0, "target": 1}
    ]"#;
    
    let nodes_val = js_sys::JSON::parse(nodes_json).expect("Failed to parse nodes JSON");
    let edges_val = js_sys::JSON::parse(edges_json).expect("Failed to parse edges JSON");
    
    // Spacing 1.5 on dist = sqrt(32) ~ 5.657 => floor(5.657 / 1.5) + 2 = 5 control points (10 coordinates)
    let state_res = SimulationState::new(width, height, nodes_val, edges_val, 1.5);
    assert!(state_res.is_ok());
    
    let mut state = state_res.unwrap();
    
    // Test initial fields updating
    state.update_physics_fields(0.05, 1.0);
    assert_eq!(state.get_potential_field().len(), 64);
    
    // Test stepping simulation
    state.step(0.1, 0.5, 0.95);
    assert_eq!(state.get_control_points().len(), 10);
    assert_eq!(state.get_control_point_counts()[0], 5);
    assert_eq!(state.get_control_point_offsets()[0], 0);
}

#[wasm_bindgen_test]
fn test_simulation_state_reset_and_update() {
    let width = 8;
    let height = 8;
    
    let nodes_json = r#"[
        {"x": 2.0, "y": 2.0, "mass": 10.0, "degree": 1.0},
        {"x": 6.0, "y": 6.0, "mass": 20.0, "degree": 2.0}
    ]"#;
    let edges_json = r#"[
        {"source": 0, "target": 1}
    ]"#;
    
    let nodes_val = js_sys::JSON::parse(nodes_json).expect("Failed to parse nodes JSON");
    let edges_val = js_sys::JSON::parse(edges_json).expect("Failed to parse edges JSON");
    
    let mut state = SimulationState::new(width, height, nodes_val, edges_val, 1.5).unwrap();
    
    // Save initial coordinates
    let initial_points = state.get_control_points();
    
    // Run simulation step on CPU
    state.update_physics_fields(0.05, 1.0);
    state.step(0.1, 0.5, 0.95);
    let stepped_points = state.get_control_points();
    
    // Points should have changed coordinates due to spring / potential forces
    assert_ne!(initial_points, stepped_points);
    
    // Test resetting returns coordinates back to straight interpolation
    state.reset_control_points();
    let reset_points = state.get_control_points();
    assert_eq!(initial_points, reset_points);
}
