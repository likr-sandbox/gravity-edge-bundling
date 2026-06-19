import init, { SimulationState } from './wasm/gravity_edge_bundling.js';

let wasmInitialized = false;
let wasmPromise = null;

export async function loadWasm() {
  if (wasmInitialized) {
    return { SimulationState };
  }
  
  if (!wasmPromise) {
    wasmPromise = init()
      .then(() => {
        wasmInitialized = true;
        return { SimulationState };
      })
      .catch((err) => {
        wasmPromise = null;
        console.error("Failed to initialize WebAssembly simulation module:", err);
        throw err;
      });
  }
  
  return wasmPromise;
}
