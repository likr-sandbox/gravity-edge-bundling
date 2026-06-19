import React, { useState, useEffect, useRef, useMemo } from 'react';
import { loadWasm } from './WasmManager';
import { loadDatasets, scaleCoordinates } from './DataLoader';
import './App.css';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [loadingMsg, setLoadingMsg] = useState('Initializing WebAssembly module...');

  // Simulation parameters
  const [gridWidth, setGridWidth] = useState(256);
  const [gridHeight, setGridHeight] = useState(256);
  const [gConstant, setGConstant] = useState(0.04);
  const [softening, setSoftening] = useState(8.0);
  const [rangeScale, setRangeScale] = useState(1.0);
  const [springK, setSpringK] = useState(0.06); // "光の速度" parameter
  const [massScale, setMassScale] = useState(15.0);
  const [controlPointSpacing, setControlPointSpacing] = useState(15.0);
  const [dt, setDt] = useState(0.6);
  const [damping, setDamping] = useState(0.95);

  // Rendering choices
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.85);
  const [showNodes, setShowNodes] = useState(true);
  const [showStraightEdges, setShowStraightEdges] = useState(false);
  const [showBundledEdges, setShowBundledEdges] = useState(true);
  const [edgeOpacity, setEdgeOpacity] = useState(0.12);

  // Simulation states
  const [isPlaying, setIsPlaying] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [stats, setStats] = useState({ nodesCount: 0, edgesCount: 0 });

  // Data holders
  const [rawAirports, setRawAirports] = useState([]);
  const [rawFlights, setRawFlights] = useState([]);
  const [scaledNodes, setScaledNodes] = useState([]);
  const [mappedEdges, setMappedEdges] = useState([]);

  // Interactive hover states
  const [hoveredNodeIndex, setHoveredNodeIndex] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  // Refs for animation and state caching
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  const simStateRef = useRef(null);
  const isPlayingRef = useRef(false);
  const stepCountRef = useRef(0);

  const paramsRef = useRef({ springK, dt, damping });
  const drawRef = useRef(null);

  // Update drawRef on every render to avoid stale closure in animation loop
  useEffect(() => {
    drawRef.current = draw;
  });

  // Create offscreen canvas once
  useEffect(() => {
    offscreenCanvasRef.current = document.createElement('canvas');
  }, []);

  // Update state refs to avoid closure stale values in animation loop
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    paramsRef.current = { springK, dt, damping };
  }, [springK, dt, damping]);

  // Load Wasm and datasets on mount
  useEffect(() => {
    const initApp = async () => {
      try {
        setLoadingMsg('Loading airport and flight datasets...');
        const { airports, flights } = await loadDatasets();

        setRawAirports(airports);
        setRawFlights(flights);
        setStats({ nodesCount: airports.length, edgesCount: flights.length });

        setLoadingMsg('Compiling Rust edge-bundling algorithms...');
        await loadWasm();

        setIsLoading(false);
      } catch (err) {
        console.error(err);
        setLoadingMsg('Initialization failed. Check console for details.');
      }
    };

    initApp();
  }, []);

  // Compute index mapping for edges once datasets are loaded
  const finalMappedEdges = useMemo(() => {
    if (rawAirports.length === 0 || rawFlights.length === 0) return [];

    const airportMap = new Map();
    rawAirports.forEach((ap, idx) => {
      airportMap.set(ap.iata, idx);
    });

    const edges = [];
    rawFlights.forEach(fl => {
      const srcIdx = airportMap.get(fl.origin);
      const dstIdx = airportMap.get(fl.destination);
      if (srcIdx !== undefined && dstIdx !== undefined) {
        edges.push({
          source: srcIdx,
          target: dstIdx,
        });
      }
    });

    return edges;
  }, [rawAirports, rawFlights]);

  // Initialize or re-create simulation when grid resolution, control points, or dataset changes
  useEffect(() => {
    if (isLoading || rawAirports.length === 0) return;

    const initializeSimulation = async () => {
      const { SimulationState } = await loadWasm();

      // Project nodes to grid coordinates [0, gridWidth] x [0, gridHeight]
      const scaled = scaleCoordinates(rawAirports, gridWidth, gridHeight, 20);
      setScaledNodes(scaled);

      // Re-create mapped edges to match scaled nodes list
      const airportIdxMap = new Map();
      scaled.forEach((ap, idx) => {
        airportIdxMap.set(ap.iata, idx);
      });

      const edges = [];
      rawFlights.forEach(fl => {
        const srcIdx = airportIdxMap.get(fl.origin);
        const dstIdx = airportIdxMap.get(fl.destination);
        if (srcIdx !== undefined && dstIdx !== undefined) {
          edges.push({
            source: srcIdx,
            target: dstIdx,
          });
        }
      });
      setMappedEdges(edges);

      if (edges.length === 0) return;

      setLoadingMsg('Initializing Rust simulation state...');
      const state = new SimulationState(gridWidth, gridHeight, scaled, edges, controlPointSpacing);

      // Splat nodes and convolve potential field in Wasm
      const nodesForWasm = scaled.map(n => ({
        x: n.x,
        y: n.y,
        mass: n.degree * massScale,
      }));
      state.update_nodes(nodesForWasm);
      state.update_physics_fields(gConstant, softening, rangeScale);

      simStateRef.current = state;
      stepCountRef.current = 0;
      setStepCount(0);
      drawRef.current?.();
    };

    initializeSimulation();
  }, [isLoading, gridWidth, gridHeight, controlPointSpacing, rawAirports]);

  // Recalculate potential fields when physical field parameters change (without resetting edge curves)
  useEffect(() => {
    const state = simStateRef.current;
    if (!state || scaledNodes.length === 0) return;

    const nodesForWasm = scaledNodes.map(n => ({
      x: n.x,
      y: n.y,
      mass: n.degree * massScale,
    }));
    state.update_nodes(nodesForWasm);
    state.update_physics_fields(gConstant, softening, rangeScale);
    drawRef.current?.();
  }, [gConstant, softening, rangeScale, massScale, scaledNodes]);

  // Animation frame loop
  useEffect(() => {
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsIntervalTime = lastTime;
    let animationId = null;

    const loop = (now) => {
      // Calculate FPS
      frameCount++;
      if (now - fpsIntervalTime >= 1000) {
        setFps(Math.round((frameCount * 1000) / (now - fpsIntervalTime)));
        frameCount = 0;
        fpsIntervalTime = now;
      }

      const state = simStateRef.current;
      if (state && isPlayingRef.current) {
        // Run a simulation step in Wasm
        const { springK, dt, damping } = paramsRef.current;
        state.step(springK, dt, damping);
        stepCountRef.current += 1;
        setStepCount(stepCountRef.current);
      }

      drawRef.current?.();
      animationId = requestAnimationFrame(loop);
    };

    if (!isLoading) {
      animationId = requestAnimationFrame(loop);
    }

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [isLoading]);

  // Main Draw function
  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear screen with premium space dark color
    ctx.fillStyle = '#060911';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const state = simStateRef.current;
    if (!state) return;

    // Render layers
    // 1. Heatmap layer
    if (showHeatmap && heatmapOpacity > 0) {
      const offscreen = offscreenCanvasRef.current;
      if (offscreen) {
        if (offscreen.width !== gridWidth || offscreen.height !== gridHeight) {
          offscreen.width = gridWidth;
          offscreen.height = gridHeight;
        }

        const offCtx = offscreen.getContext('2d');
        const potential = state.get_potential_field();

        if (potential && potential.length > 0 && offCtx) {
          let minPot = Infinity;
          let maxPot = -Infinity;
          for (let i = 0; i < potential.length; i++) {
            const v = potential[i];
            if (v < minPot) minPot = v;
            if (v > maxPot) maxPot = v;
          }

          const imgData = offCtx.createImageData(gridWidth, gridHeight);
          const data = imgData.data;
          const range = maxPot - minPot;

          for (let i = 0; i < potential.length; i++) {
            const val = potential[i];
            // Normalize: potential is most negative near nodes
            const intensity = range > 0.0001 ? (maxPot - val) / range : 0.0;
            const idx = i * 4;

            if (intensity > 0.02) {
              const t = intensity;
              // Radiant deep violet to bright cyan gradient glow
              data[idx] = Math.floor(t * t * 140); // R
              data[idx + 1] = Math.floor(t * 180 + (t > 0.6 ? (t - 0.6) * 150 : 0)); // G
              data[idx + 2] = Math.floor(80 + t * 175); // B
              data[idx + 3] = Math.floor(t * 230 * heatmapOpacity); // A
            } else {
              data[idx + 3] = 0;
            }
          }
          offCtx.putImageData(imgData, 0, 0);

          // Draw offscreen grid stretched smoothly onto visual canvas
          ctx.imageSmoothingEnabled = true;
          ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
        }
      }
    }

    const renderScale = canvas.width / gridWidth;

    // 2. Straight Edges layer (underlay)
    if (showStraightEdges && scaledNodes.length > 0) {
      ctx.lineWidth = 0.5;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';

      mappedEdges.forEach(edge => {
        const src = scaledNodes[edge.source];
        const dst = scaledNodes[edge.target];
        if (src && dst) {
          ctx.beginPath();
          ctx.moveTo(src.x * renderScale, src.y * renderScale);
          ctx.lineTo(dst.x * renderScale, dst.y * renderScale);
          ctx.stroke();
        }
      });
    }

    // 3. Bundled Edges layer
    if (showBundledEdges && mappedEdges.length > 0) {
      const cpData = state.get_control_points();
      const offsets = state.get_control_point_offsets();
      const counts = state.get_control_point_counts();

      if (cpData && offsets && counts) {
        for (let e_idx = 0; e_idx < mappedEdges.length; e_idx++) {
          const edge = mappedEdges[e_idx];
          const isHighlighted = hoveredNodeIndex !== null &&
            (edge.source === hoveredNodeIndex || edge.target === hoveredNodeIndex);

          if (isHighlighted) {
            ctx.lineWidth = 2.0;
            ctx.strokeStyle = 'rgba(0, 255, 200, 0.8)';
            ctx.shadowColor = 'rgba(0, 255, 200, 0.5)';
            ctx.shadowBlur = 6;
          } else {
            ctx.lineWidth = 1.0;
            ctx.strokeStyle = `rgba(0, 210, 255, ${edgeOpacity})`;
            ctx.shadowBlur = 0;
          }

          const offset = offsets[e_idx];
          const count = counts[e_idx];
          if (offset !== undefined && count !== undefined) {
            ctx.beginPath();
            const startBase = 2 * offset;
            ctx.moveTo(cpData[startBase] * renderScale, cpData[startBase + 1] * renderScale);

            for (let i = 1; i < count; i++) {
              const base = 2 * (offset + i);
              ctx.lineTo(cpData[base] * renderScale, cpData[base + 1] * renderScale);
            }
            ctx.stroke();
          }
        }
        ctx.shadowBlur = 0; // Reset shadow
      }
    }

    // 4. Nodes layer
    if (showNodes && scaledNodes.length > 0) {
      scaledNodes.forEach((node, idx) => {
        const isHovered = hoveredNodeIndex === idx;

        // Logarithmic node radius based on degree
        const r = Math.max(2, Math.log2(node.degree + 1) * 1.3);

        ctx.beginPath();
        ctx.arc(node.x * renderScale, node.y * renderScale, isHovered ? r + 3 : r, 0, 2 * Math.PI);

        if (isHovered) {
          ctx.fillStyle = '#00ffc8';
          ctx.shadowColor = '#00ffc8';
          ctx.shadowBlur = 12;
        } else {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
          ctx.shadowBlur = 0;
        }
        ctx.fill();
      });
      ctx.shadowBlur = 0; // Reset shadow
    }
  };

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || scaledNodes.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const mouseY = ((e.clientY - rect.top) / rect.height) * canvas.height;

    const renderScale = canvas.width / gridWidth;
    let closestIdx = null;
    let minDist = 14;

    scaledNodes.forEach((node, idx) => {
      const nx = node.x * renderScale;
      const ny = node.y * renderScale;
      const dist = Math.hypot(mouseX - nx, mouseY - ny);
      if (dist < minDist) {
        minDist = dist;
        closestIdx = idx;
      }
    });

    if (closestIdx !== null) {
      setHoveredNodeIndex(closestIdx);
      setTooltipPos({
        x: e.clientX - rect.left + 15,
        y: e.clientY - rect.top + 15,
      });
    } else {
      setHoveredNodeIndex(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredNodeIndex(null);
  };

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
  };

  const stepSimulation = () => {
    const state = simStateRef.current;
    if (state) {
      state.step(springK, dt, damping);
      stepCountRef.current += 1;
      setStepCount(stepCountRef.current);
      drawRef.current?.();
    }
  };

  const resetSimulation = () => {
    const state = simStateRef.current;
    if (state) {
      state.reset_control_points();
      stepCountRef.current = 0;
      setStepCount(0);
      drawRef.current?.();
    }
  };

  const hoveredNode = hoveredNodeIndex !== null ? scaledNodes[hoveredNodeIndex] : null;

  return (
    <div className="app-container">
      {isLoading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <div className="loading-text">{loadingMsg}</div>
        </div>
      )}

      <div className="sidebar">
        <div className="sidebar-header">
          <h1>⚡ GRAVITY BUNDLING</h1>
          <p>FFT-Based Gravitational Edge Bundling</p>
        </div>

        <div className="sidebar-content">
          {/* Controls Panel */}
          <div>
            <div className="section-title">Simulation Loop</div>
            <div className="button-row">
              <button
                onClick={togglePlayback}
                className={`primary-btn ${isPlaying ? 'playing' : ''}`}
              >
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>
              <button onClick={stepSimulation} disabled={isPlaying}>
                Step
              </button>
              <button onClick={resetSimulation}>
                Reset
              </button>
            </div>
          </div>

          <div>
            <div className="section-title">Physical Constants</div>
            <div className="control-group">
              <div className="control-item">
                <div className="control-label">
                  <span>万有引力定数 G (Gravity)</span>
                  <span className="control-value">{gConstant.toFixed(4)}</span>
                </div>
                <input
                  type="range"
                  min="0.0001"
                  max="0.15"
                  step="0.0001"
                  value={gConstant}
                  onChange={(e) => setGConstant(parseFloat(e.target.value))}
                />
              </div>

              <div className="control-item">
                <div className="control-label">
                  <span>ソフトニング定数 ε</span>
                  <span className="control-value">{softening.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="35.0"
                  step="0.5"
                  value={softening}
                  onChange={(e) => setSoftening(parseFloat(e.target.value))}
                />
              </div>

              <div className="control-item">
                <div className="control-label">
                  <span>重力影響範囲 (Range Scale)</span>
                  <span className="control-value">{rangeScale.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="3.0"
                  step="0.01"
                  value={rangeScale}
                  onChange={(e) => setRangeScale(parseFloat(e.target.value))}
                />
              </div>

              <div className="control-item">
                <div className="control-label">
                  <span>光の速度 (バネ定数 k)</span>
                  <span className="control-value">{springK.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.5"
                  step="0.005"
                  value={springK}
                  onChange={(e) => setSpringK(parseFloat(e.target.value))}
                />
              </div>

              <div className="control-item">
                <div className="control-label">
                  <span>ノードの質量係数</span>
                  <span className="control-value">{massScale.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="50.0"
                  step="1.0"
                  value={massScale}
                  onChange={(e) => setMassScale(parseFloat(e.target.value))}
                />
              </div>
            </div>
          </div>

          <div>
            <div className="section-title">Numerical Grid & Solver</div>
            <div className="control-group">
              <div className="control-item">
                <div className="control-label">
                  <span>グリッド解像度 (W x H)</span>
                </div>
                <select
                  value={gridWidth}
                  onChange={(e) => {
                    const size = parseInt(e.target.value);
                    setGridWidth(size);
                    setGridHeight(size);
                  }}
                >
                  <option value="128">128 x 128</option>
                  <option value="256">256 x 256</option>
                  <option value="512">512 x 512</option>
                  <option value="1024">1024 x 1024</option>
                </select>
              </div>

              <div className="control-item">
                <div className="control-label">
                  <span>制御点の間隔 (Grid Spacing)</span>
                  <span className="control-value">{controlPointSpacing.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="5.0"
                  max="50.0"
                  step="1.0"
                  value={controlPointSpacing}
                  onChange={(e) => setControlPointSpacing(parseFloat(e.target.value))}
                />
              </div>

              <div className="control-item">
                <div className="control-label">
                  <span>時間ステップ dt</span>
                  <span className="control-value">{dt.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.05"
                  value={dt}
                  onChange={(e) => setDt(parseFloat(e.target.value))}
                />
              </div>

              <div className="control-item">
                <div className="control-label">
                  <span>ダンピング係数</span>
                  <span className="control-value">{damping.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.80"
                  max="1.00"
                  step="0.01"
                  value={damping}
                  onChange={(e) => setDamping(parseFloat(e.target.value))}
                />
              </div>
            </div>
          </div>

          <div>
            <div className="section-title">Visual Overlays</div>
            <div className="control-group">
              <div className="toggle-item">
                <span className="toggle-label">重力ポテンシャル場 (Heatmap)</span>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={showHeatmap}
                    onChange={(e) => setShowHeatmap(e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>

              {showHeatmap && (
                <div className="control-item">
                  <div className="control-label">
                    <span>ヒートマップ不透明度</span>
                    <span className="control-value">{Math.round(heatmapOpacity * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={heatmapOpacity}
                    onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                  />
                </div>
              )}

              <div className="toggle-item">
                <span className="toggle-label">ノード (Airports)</span>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={showNodes}
                    onChange={(e) => setShowNodes(e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>

              <div className="toggle-item">
                <span className="toggle-label">エッジ (Bundled Routes)</span>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={showBundledEdges}
                    onChange={(e) => setShowBundledEdges(e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>

              {showBundledEdges && (
                <div className="control-item">
                  <div className="control-label">
                    <span>ルート不透明度</span>
                    <span className="control-value">{Math.round(edgeOpacity * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.4"
                    step="0.01"
                    value={edgeOpacity}
                    onChange={(e) => setEdgeOpacity(parseFloat(e.target.value))}
                  />
                </div>
              )}

              <div className="toggle-item">
                <span className="toggle-label">直線ルート (Original)</span>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={showStraightEdges}
                    onChange={(e) => setShowStraightEdges(e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="visualizer-container">
        <div className="hud-panel">
          <span className="hud-label">Sim Step:</span>
          <span className="hud-val">{stepCount}</span>
          <span className="hud-label">FPS:</span>
          <span className="hud-val">{fps}</span>
          <span className="hud-label">Airports:</span>
          <span className="hud-val">{stats.nodesCount}</span>
          <span className="hud-label">Routes:</span>
          <span className="hud-val">{stats.edgesCount}</span>
        </div>

        <div className="canvas-wrapper">
          <canvas
            ref={canvasRef}
            width={850}
            height={850}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />

          {hoveredNode && (
            <div
              className="node-tooltip"
              style={{
                left: `${tooltipPos.x}px`,
                top: `${tooltipPos.y}px`,
              }}
            >
              <div className="tooltip-title">{hoveredNode.name}</div>
              <div className="tooltip-detail">
                <span className="tooltip-label">Code (IATA)</span>
                <span className="tooltip-val">{hoveredNode.iata}</span>
              </div>
              <div className="tooltip-detail">
                <span className="tooltip-label">City</span>
                <span className="tooltip-val">{hoveredNode.city}, {hoveredNode.state}</span>
              </div>
              <div className="tooltip-detail">
                <span className="tooltip-label">Routes</span>
                <span className="tooltip-val">{hoveredNode.degree} flights</span>
              </div>
              <div className="tooltip-detail">
                <span className="tooltip-label">Total Volume</span>
                <span className="tooltip-val">{hoveredNode.flightCount}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
