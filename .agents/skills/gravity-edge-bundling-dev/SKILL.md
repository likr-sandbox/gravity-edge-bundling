---
name: gravity-edge-bundling-dev
description: Developer guide for maintaining, testing, updating, and optimizing the gravity-edge-bundling repository. Make sure to use this skill whenever the user mentions updating edge bundling logic, modifying Rust/Wasm physics simulation, changing React visualizer controls, running tests, or performing any code refactoring in this repository.
---

# Gravity Edge Bundling Developer Guide

This guide details the architecture, mathematical models, Wasm-JS data bridge, and development workflows for the FFT-based Gravity Edge Bundling application. Refer to this skill when making any modifications to the codebase to maintain design integrity and prevent regression.

---

## 1. Directory Structure & Architecture

The project consists of a hybrid Rust (WebAssembly) backend and a React (Vite) frontend:

- **`gravity-edge-bundling/`**: Rust Crate (compiled to WebAssembly)
  - [`src/lib.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/src/lib.rs): Thin Wasm wrapper (`SimulationState`) delegating directly to the pure Rust core.
  - [`src/simulation.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/src/simulation.rs): Core physics simulation containing grid splatting, FFT convolution, force fields, and spring-step calculations.
  - [`src/fft2d.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/src/fft2d.rs): 2D Fast Fourier Transform helper.
  - [`tests/web.rs`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/gravity-edge-bundling/tests/web.rs): Wasm integration tests executed via Node.js.
- **`src/`**: React Visualizer Frontend
  - [`src/App.jsx`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/src/App.jsx): Canvas-based renderer and interactive sidebar controls.
  - [`src/WasmManager.js`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/src/WasmManager.js): Async loader for the WebAssembly module.
  - [`src/DataLoader.js`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/src/DataLoader.js): CSV loader for airport and flight data, coordinates projection.
  - [`src/wasm/`](file:///home/likr/work/likr-sandbox/gravity-edge-bundling/src/wasm): Generated WebAssembly binaries and bindings.

---

## 2. Mathematical Models & Physics Formulation

To maintain stable simulations, adhere to these mathematical formulations:

### A. Repulsive Gravity Field
To prevent control points from collapsing into node endpoints, the potential field generates a **repulsive** force.
- **Potential Field $\Phi$**: Computed via 2D FFT convolution of the splatted node masses with a gravitational decay kernel.
- **Repulsive Force $\vec{F}$**: Computed as the positive central-difference gradient of the potential field (opposite to attractive gravity):
  $$\vec{F} = \nabla \Phi$$
  In code:
  $$F_x(x, y) = \frac{\Phi(x+1, y) - \Phi(x-1, y)}{2}$$
  $$F_y(x, y) = \frac{\Phi(x, y+1) - \Phi(x, y-1)}{2}$$

### B. Coordinate Range Scaling ($\lambda$)
To stretch or compress the reach of the potential field, coordinates are scaled inside the kernel generation:
$$dx_{\text{scaled}} = \frac{dx}{\lambda}, \quad dy_{\text{scaled}} = \frac{dy}{\lambda}$$
$$K(x, y) = - \frac{G}{\sqrt{dx_{\text{scaled}}^2 + dy_{\text{scaled}}^2 + \epsilon^2}}$$
- $\lambda = 1.0$: Default distance.
- $\lambda > 1.0$: Broad potential field (global bundling).
- $\lambda < 1.0$: Narrow potential field (localized bundling).

### C. Length-based Variable Control Points
Each edge $e$ has a variable number of control points $N_e$ determined by its original length $L_e$ and the spacing parameter $d$:
$$N_e = \max\left(3, \left\lfloor \frac{L_e}{d} \right\rfloor + 2\right)$$
This guarantees a minimum of 3 points (endpoints + 1 midpoint), ensuring stable simulation.

---

## 3. High-Performance Wasm-JS Data Bridge

To avoid JSON/Serde serialization overhead (which degrades performance at 60fps), the Wasm-JS boundary shares layout metadata using parallel flat arrays:

1. **Coordinates**: `get_control_points() -> Vec<f32>` returns a flat array of alternating `[x, y]` coordinates.
2. **Offsets**: `get_control_point_offsets() -> Vec<u32>` returns the starting index of each edge in the coordinate array.
3. **Counts**: `get_control_point_counts() -> Vec<u32>` returns the number of control points ($N_e$) for each edge.

### Drawing Loop in React:
```javascript
const cpData = state.get_control_points();
const offsets = state.get_control_point_offsets();
const counts = state.get_control_point_counts();

for (let e_idx = 0; e_idx < mappedEdges.length; e_idx++) {
  const offset = offsets[e_idx];
  const count = counts[e_idx];
  
  ctx.beginPath();
  ctx.moveTo(cpData[offset * 2] * scale, cpData[offset * 2 + 1] * scale);
  
  for (let i = 1; i < count; i++) {
    const base = (offset + i) * 2;
    ctx.lineTo(cpData[base] * scale, cpData[base + 1] * scale);
  }
  ctx.stroke();
}
```

---

## 4. React Rendering & Animation Loop Guidelines

When updating the React frontend, prevent stale closures inside the `requestAnimationFrame` loop by using `useRef` to store state updates:

- **State Caching**: Use Refs like `isPlayingRef` and `paramsRef` to cache physical values (`springK`, `dt`, `damping`) and run states.
- **Rendering Caching**: Store the draw function itself in a ref (`drawRef.current = draw`) so the animation loop always executes the latest closure.

---

## 5. Development & Testing Commands

Before committing any modifications, verify build and test compliance.

### A. Run Pure Rust Unit Tests
Always keep simulation math testable on host CPU.
```bash
cargo test --manifest-path gravity-edge-bundling/Cargo.toml
```

### B. Run Wasm Node.js Integration Tests
Verify the JS-to-Wasm bridge under a Node.js test environment.
```bash
npx wasm-pack test --node gravity-edge-bundling
```

### C. Compile WebAssembly Module
Compile and output Wasm bindings directly into the React source folder.
```bash
npx wasm-pack build --target web --out-dir ../src/wasm gravity-edge-bundling
```

### D. Build Production Client Bundle
Verify Vite compiles and packages everything successfully.
```bash
npm run build
```
