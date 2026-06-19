---
name: gravity-edge-bundling-dev
description: Developer guide for maintaining, testing, updating, and optimizing the gravity-edge-bundling repository. Make sure to use this skill whenever the user mentions updating edge bundling logic, modifying Rust/Wasm physics simulation, changing React visualizer controls, running tests, or performing any code refactoring in this repository.
---

# Gravity Edge Bundling Developer Guide

This guide details the architecture, mathematical models, WebGPU acceleration, Wasm-JS data bridge, and development workflows for the FFT-based Gravity Edge Bundling application. Refer to this skill when making any modifications to the codebase to maintain design integrity and prevent regression.

---

## 1. Directory Structure & Architecture

The project consists of a hybrid Rust (WebAssembly) backend and a React (Vite) frontend:

- **`gravity-edge-bundling/`**: Rust Crate (compiled to WebAssembly / Native host)
  - [`src/lib.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/src/lib.rs): Thin Wasm wrapper and `SimulationState` class orchestrating CPU & GPU pipelines.
  - [`src/simulation.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/src/simulation.rs): CPU physics simulation (mass splatting, FFT 2D convolution, metadata generation).
  - [`src/webgpu.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/src/webgpu.rs): WebGPU pipeline logic containing storage/uniform buffer managers, textures, compute/render pipelines, and WGSL shaders.
  - [`src/fft2d.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/src/fft2d.rs): 2D Fast Fourier Transform helper.
  - [`tests/webgpu_test.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/tests/webgpu_test.rs): Native WebGPU compute integration test executed on the host GPU.
  - [`tests/web.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/tests/web.rs): Wasm integration tests executed via Node.js (CPU mode).
- **`src/`**: React Visualizer Frontend
  - [`src/App.jsx`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/src/App.jsx): Main visualizer frontend passing Canvas to the Wasm initializer and calling render steps.
  - [`src/WasmManager.js`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/src/WasmManager.js): Async loader for the WebAssembly module.
  - [`src/wasm/`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/src/wasm): Generated WebAssembly binaries and bindings.

---

## 2. Physics Simulation & WebGPU Acceleration

The physics simulation is hybrid: the CPU handles global grid mass splatting and FFT convolution, while the GPU handles control point force updates and rendering.

### A. Repulsive Gravity & Spring Forces
- **Spring Forces**: Computed in the compute shader using adjacent control point coordinates:
  $$\vec{F}_{\text{spring}} = K_{\text{spring}} \cdot (\vec{P}_{\text{prev}} + \vec{P}_{\text{next}} - 2\vec{P}_{\text{curr}})$$
- **Gravity Forces**: Computed in the compute shader by sampling the uploaded force field texture at the control point's position.

### B. Double-Buffered Compute Pipeline
Control point coordinates are double-buffered (`positions_a` and `positions_b`) to prevent race conditions during parallel shader execution:
1. The compute shader reads coordinates from the source buffer and writes updated coordinates to the destination buffer.
2. The roles of the source and destination buffers are swapped at the end of the physics step.

### C. Manual Bilinear Sampling (Broad Device Compatibility)
- **Problem**: Standard WebGPU classifies 32-bit float formats (`R32Float` and `Rg32Float`) as `UnfilterableFloat` by default. Hardware-level bilinear filtering on these formats is optional and requires the `float32-filterable` device extension, which is missing on many devices.
- **Solution**: To guarantee 100% device compatibility, use a **Nearest** sampler and **manual bilinear interpolation** in the WGSL shaders using `textureLoad`:
  ```wgsl
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
  ```

---

## 3. WebGPU Rendering Pipeline Layout

To prevent WebGPU validation errors, the render pipelines use a layout partitioned into two bind groups:

- **Group 0 (`render_params_layout`)**: Expects `render_params_bind_group`, containing only the uniform `Params` buffer (`binding: 0`).
- **Group 1 (`render_resources_layout`)**: Expects `render_bind_group`, containing textures, samplers, and storage buffers for vertex/fragment rendering.

### Render Draw Loop
During rendering, configure the pipelines and bind groups sequentially:
```rust
render_pass.set_pipeline(heatmap_pipeline);
render_pass.set_bind_group(0, &self.render_params_bind_group, &[]);
render_pass.set_bind_group(1, &self.render_bind_group, &[]);
render_pass.draw(0..6, 0..1);
```

---

## 4. Headless Execution & Command-Line Testing

To make WebGPU testable on native host platforms without a display/window:
1. **`WgpuContext::new_headless`**: Instantiates a GPU context with `surface`, `config`, and rendering pipelines set to `None`.
2. **Readback Staging Buffers**: `WgpuContext::read_positions` copies positions to a staging buffer with `MAP_READ` usage and maps it using a standard channel:
   ```rust
   let (tx, rx) = std::sync::mpsc::channel();
   buffer_slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
   self.device.poll(wgpu::Maintain::Wait);
   rx.recv().unwrap().unwrap();
   ```

---

## 5. Development & Testing Commands

Always verify builds and tests before compiling binding modules.

### A. Run Native Host Tests (CPU and GPU Headless)
Verify the WebGPU compute shader compiles and moves points correctly on the local GPU:
```bash
cargo test --test webgpu_test --manifest-path gravity-edge-bundling/Cargo.toml -- --nocapture
```
For general CPU tests:
```bash
cargo test --manifest-path gravity-edge-bundling/Cargo.toml
```

### B. Compile WebAssembly Module
Compile the Rust crate to Wasm and output bindings into the React source folder:
```bash
npx wasm-pack build --target web --out-dir ../src/wasm gravity-edge-bundling
```

### C. Run Local Visualizer
Start the Vite development server to launch the visualizer:
```bash
npm run dev
```
*(Note: If changes to the Wasm bindings are not reflected in the browser, delete `.vite/deps` cache and force a hard reload of the page).*
