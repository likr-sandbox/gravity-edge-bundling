use gravity_edge_bundling::simulation::{Node, Edge};
use gravity_edge_bundling::SimulationState;

#[tokio::test]
async fn test_webgpu_compute_step() {
    let width = 8;
    let height = 8;
    
    let nodes = vec![
        Node { x: 2.0, y: 2.0, mass: 10.0, degree: 1.0 },
        Node { x: 6.0, y: 6.0, mass: 20.0, degree: 2.0 },
    ];
    let edges = vec![
        Edge { source: 0, target: 1 },
    ];
    
    let mut state = SimulationState::new_native(width, height, nodes, edges, 1.5);
    
    // Save initial coordinates on CPU
    let initial_points = state.get_control_points();
    assert_eq!(initial_points.len(), 10); // 5 control points = 10 coordinates
    
    // Initialize WebGPU headless context
    let init_res = state.init_wgpu_headless().await;
    assert!(init_res.is_ok(), "Failed to initialize headless WebGPU: {:?}", init_res.err());
    
    // Upload physics fields
    state.update_physics_fields(0.05, 1.0, 0.0);
    
    // Run simulation step on GPU
    state.step(0.1, 0.5, 0.95);
    
    // Read back control points from the GPU
    let gpu_points_res = state.get_control_points_gpu().await;
    assert!(gpu_points_res.is_ok(), "Failed to read positions from GPU: {:?}", gpu_points_res.err());
    
    let gpu_points = gpu_points_res.unwrap();
    assert_eq!(gpu_points.len(), 10);
    
    // The control points should have changed coordinates
    assert_ne!(initial_points, gpu_points);
    
    println!("Initial CPU points: {:?}", initial_points);
    println!("Stepped GPU points: {:?}", gpu_points);
}
