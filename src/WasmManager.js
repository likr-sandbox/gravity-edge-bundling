import init, { SimulationState, create_simulation_state } from './wasm/gravity_edge_bundling.js';

// WebGPU limits compatibility patch for browser engines that don't support new limits
if (typeof GPUAdapter !== 'undefined' && GPUAdapter.prototype.requestDevice) {
  const originalRequestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = function(descriptor) {
    if (descriptor && descriptor.requiredLimits) {
      delete descriptor.requiredLimits.maxInterStageShaderComponents;
      delete descriptor.requiredLimits.maxInterStageShaderVariables;
    }
    return originalRequestDevice.call(this, descriptor);
  };
}

let wasmInitialized = false;
let wasmPromise = null;

export async function loadWasm() {
  if (wasmInitialized) {
    return { SimulationState, create_simulation_state };
  }
  
  if (!wasmPromise) {
    wasmPromise = init()
      .then(() => {
        wasmInitialized = true;
        return { SimulationState, create_simulation_state };
      })
      .catch((err) => {
        wasmPromise = null;
        console.error("Failed to initialize WebAssembly simulation module:", err);
        throw err;
      });
  }
  
  return wasmPromise;
}
