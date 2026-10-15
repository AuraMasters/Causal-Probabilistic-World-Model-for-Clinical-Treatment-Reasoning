import { useMemo, useState, useCallback } from 'react'
import dagre from 'dagre'
import { AnimatePresence, motion } from 'framer-motion'
import {
  GitBranch,
  Info,
  Maximize2,
  Minimize2,
  RefreshCw,
  X,
} from 'lucide-react'
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  useReactFlow,
  ReactFlowProvider,
  type Edge as FlowEdge,
  type Node as FlowNode,
  type NodeProps,
  type NodeTypes,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import type { Overview } from '../lib/types'

const NODE_DESCRIPTIONS: Record<string, { name: string; category: string; desc: string }> = {
  age: { name: 'Age', category: 'Demographics', desc: 'Patient age at study entry' },
  wtkg: { name: 'Weight (kg)', category: 'Demographics', desc: 'Baseline body weight in kilograms' },
  hemo: { name: 'Hemophilia', category: 'Clinical History', desc: 'Presence of hemophilia disorder' },
  homo: { name: 'Homosexuality', category: 'Demographics', desc: 'Homosexual transmission risk factor' },
  drugs: { name: 'IV Drug Use', category: 'Clinical History', desc: 'History of intravenous drug abuse' },
  karnof: { name: 'Karnofsky Score', category: 'Functional Status', desc: 'Performance scale rating (0-100)' },
  oprior: { name: 'Prior Opportunistic Infection', category: 'Clinical History', desc: 'Pre-trial opportunistic disease history' },
  z30: { name: 'Prior ZDV (30 Days)', category: 'Treatment History', desc: 'Zidovudine use in the 30 days prior to study' },
  preanti: { name: 'Pre-ART Exposure (Days)', category: 'Treatment History', desc: 'Cumulative days of antiretroviral exposure' },
  race: { name: 'Race', category: 'Demographics', desc: 'Self-reported race classification' },
  gender: { name: 'Gender', category: 'Demographics', desc: 'Biological sex' },
  strat: { name: 'Antiretroviral Stratum', category: 'Treatment History', desc: 'Stratification category based on prior ART duration' },
  symptom: { name: 'Symptomatic Indicator', category: 'Clinical Status', desc: 'Presence of HIV symptoms at baseline' },
  cd40: { name: 'Baseline CD4 T-Cell Count', category: 'Biomarker', desc: 'Baseline helper T-cell count (cells/mm³)' },
  cd80: { name: 'Baseline CD8 T-Cell Count', category: 'Biomarker', desc: 'Baseline cytotoxic T-cell count (cells/mm³)' },
  trt: { name: 'Treatment Regimen (Intervention)', category: 'Intervention', desc: 'Randomized treatment arm (4 options)' },
  label: { name: 'Primary Trial Outcome (Label)', category: 'Outcome', desc: '50% CD4 decline, AIDS endpoint, or death' },
}

type DagNodeData = {
  label: string
  name: string
  category: string
  dimmed: boolean
  isTarget: boolean
  isSource: boolean
}

function DagNode({ data, selected }: NodeProps) {
  const { label, name, category, dimmed, isTarget, isSource } = data as DagNodeData

  return (
    <div
      className={`relative w-[170px] rounded-xl border bg-zinc-950 p-3 shadow-md backdrop-blur transition-all duration-150 ${
        selected
          ? 'border-white ring-2 ring-white bg-zinc-900 scale-105'
          : isSource || isTarget
            ? 'border-zinc-400 bg-zinc-900'
            : 'border-zinc-800'
      } ${dimmed ? 'opacity-30' : 'opacity-100'}`}
    >
      <Handle type="target" position={Position.Left} className="!h-2 !w-2 !border-0 !bg-white" />
      <div className="flex items-center justify-between">
        <span className="rounded bg-zinc-800 border border-zinc-700 px-1.5 py-0.5 font-mono text-[11px] font-bold text-white">
          {label}
        </span>
        <span className="font-mono text-[9px] text-zinc-500 uppercase">{category}</span>
      </div>
      <div className="mt-1 font-display text-xs font-bold text-white truncate" title={name}>
        {name}
      </div>
      <Handle type="source" position={Position.Right} className="!h-2 !w-2 !border-0 !bg-white" />
    </div>
  )
}

const nodeTypes: NodeTypes = {
  customNode: DagNode,
}

export function DagGraph({ overview }: { overview: Overview }) {
  return (
    <ReactFlowProvider>
      <DagGraphInner overview={overview} />
    </ReactFlowProvider>
  )
}

function DagGraphInner({ overview }: { overview: Overview }) {
  const reactFlow = useReactFlow()
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [activeFilter, setActiveFilter] = useState<'all' | 'intervention' | 'biomarkers' | 'stable'>('all')

  const highlight = selectedNode || hoveredNode

  // Layout with Dagre
  const { rawNodes, rawEdges } = useMemo(() => {
    const g = new dagre.graphlib.Graph()
    g.setGraph({ rankdir: 'LR', nodesep: 35, ranksep: 75 })
    g.setDefaultEdgeLabel(() => ({}))

    const allNodeIds = Array.from(
      new Set([
        ...overview.variables.numerical,
        ...overview.variables.categorical,
        'trt',
        'label',
      ])
    )

    allNodeIds.forEach((id) => {
      g.setNode(id, { width: 170, height: 60 })
    })

    overview.dag.forEach((edge) => {
      g.setEdge(edge.source, edge.target)
    })

    dagre.layout(g)

    const nodes: FlowNode[] = allNodeIds.map((id) => {
      const nodeMeta = NODE_DESCRIPTIONS[id] || { name: id, category: 'Clinical', desc: '' }
      const nodePos = g.node(id)
      return {
        id,
        type: 'customNode',
        position: { x: nodePos.x - 85, y: nodePos.y - 30 },
        data: {
          label: id,
          name: nodeMeta.name,
          category: nodeMeta.category,
          dimmed: false,
          isTarget: false,
          isSource: false,
        },
      }
    })

    const edges: FlowEdge[] = overview.dag.map((edge, idx) => {
      const isIntervention = edge.is_intervention_edge || (edge.source === 'trt' && edge.target === 'label')
      const stability = edge.bootstrap_stability ?? (isIntervention ? 1.0 : 0.5)
      const isVeryStable = stability >= 0.75
      const isWeak = stability < 0.50 && !isIntervention

      return {
        id: `e-${edge.source}-${edge.target}-${idx}`,
        source: edge.source,
        target: edge.target,
        type: 'smoothstep',
        data: {
          stability,
          category: edge.support_category || (isIntervention ? 'INTERVENTION' : 'EXPLORATORY'),
          isIntervention,
          isVeryStable,
          isWeak,
        },
      }
    })

    return { rawNodes: nodes, rawEdges: edges }
  }, [overview])

  const renderNodes = useMemo(() => {
    return rawNodes.map((node) => {
      const isSelected = highlight === node.id
      const isParent = highlight ? rawEdges.some((e) => e.source === node.id && e.target === highlight) : false
      const isChild = highlight ? rawEdges.some((e) => e.source === highlight && e.target === node.id) : false

      let dimmed = false
      if (highlight) {
        dimmed = !isSelected && !isParent && !isChild
      } else if (activeFilter === 'intervention') {
        dimmed = node.id !== 'trt' && node.id !== 'label'
      } else if (activeFilter === 'biomarkers') {
        dimmed = !['cd40', 'cd80', 'karnof', 'preanti', 'z30', 'trt', 'label'].includes(node.id)
      }

      return {
        ...node,
        data: {
          ...node.data,
          dimmed,
          isSource: isParent,
          isTarget: isChild,
        },
      }
    })
  }, [rawNodes, rawEdges, highlight, activeFilter])

  const renderEdges = useMemo(() => {
    return rawEdges.map((edge) => {
      const isHighlighted = highlight !== null && (edge.source === highlight || edge.target === highlight)
      const data = edge.data as { stability: number; isIntervention: boolean; isVeryStable: boolean; isWeak: boolean }

      let strokeColor = '#52525b'
      let strokeDash = undefined

      if (data.isIntervention) {
        strokeColor = '#ffffff'
      } else if (data.isVeryStable) {
        strokeColor = isHighlighted ? '#ffffff' : '#a1a1aa'
      } else if (data.isWeak) {
        strokeColor = isHighlighted ? '#ffffff' : '#27272a'
        strokeDash = '4 4'
      } else {
        strokeColor = isHighlighted ? '#ffffff' : '#71717a'
      }

      return {
        ...edge,
        animated: isHighlighted || data.isIntervention,
        style: {
          stroke: strokeColor,
          strokeWidth: isHighlighted ? 2.5 : data.isIntervention ? 2.2 : 1.4,
          strokeDasharray: strokeDash,
        },
      }
    })
  }, [rawEdges, highlight])

  const selectedNodeMeta = selectedNode ? NODE_DESCRIPTIONS[selectedNode] : null
  const selectedStates = selectedNode ? overview.states[selectedNode] ?? [] : []
  const incomingEdges = selectedNode ? overview.dag.filter((e) => e.target === selectedNode) : []
  const outgoingEdges = selectedNode ? overview.dag.filter((e) => e.source === selectedNode) : []

  const handleFitView = useCallback(() => {
    reactFlow.fitView({ padding: 0.18, duration: 400 })
  }, [reactFlow])

  const handleZoomIn = useCallback(() => {
    reactFlow.zoomIn({ duration: 300 })
  }, [reactFlow])

  const handleZoomOut = useCallback(() => {
    reactFlow.zoomOut({ duration: 300 })
  }, [reactFlow])

  const handleFocusTrtPath = useCallback(() => {
    setSelectedNode('trt')
    reactFlow.fitView({ padding: 0.25, duration: 400 })
  }, [reactFlow])

  return (
    <div className="space-y-4">
      {/* Top Filter and Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-zinc-800 bg-zinc-950/90 p-3 shadow-md">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="font-mono text-xs font-bold text-zinc-400 uppercase mr-2 flex items-center gap-1.5">
            <GitBranch className="h-4 w-4 text-zinc-300" /> Focus View:
          </span>
          <button
            onClick={() => {
              setActiveFilter('all')
              setSelectedNode(null)
            }}
            className={`rounded-md px-3 py-1 text-xs font-semibold transition-all cursor-pointer ${
              activeFilter === 'all' && !selectedNode
                ? 'bg-white text-black shadow-sm'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            All 17 Nodes
          </button>
          <button
            onClick={handleFocusTrtPath}
            className={`rounded-md px-3 py-1 text-xs font-semibold transition-all cursor-pointer ${
              selectedNode === 'trt'
                ? 'bg-white text-black shadow-sm'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            Treatment Pathway (trt &rarr; label)
          </button>
          <button
            onClick={() => {
              setActiveFilter('biomarkers')
              setSelectedNode('cd40')
            }}
            className={`rounded-md px-3 py-1 text-xs font-semibold transition-all cursor-pointer ${
              activeFilter === 'biomarkers'
                ? 'bg-white text-black shadow-sm'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            Key Biomarkers
          </button>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomIn}
            className="rounded-md border border-zinc-800 bg-zinc-900 p-1.5 text-zinc-300 hover:text-white hover:border-zinc-700 cursor-pointer"
            title="Zoom In"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleZoomOut}
            className="rounded-md border border-zinc-800 bg-zinc-900 p-1.5 text-zinc-300 hover:text-white hover:border-zinc-700 cursor-pointer"
            title="Zoom Out"
          >
            <Minimize2 className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleFitView}
            className="rounded-md border border-zinc-800 bg-zinc-900 p-1.5 text-zinc-300 hover:text-white hover:border-zinc-700 cursor-pointer"
            title="Fit View"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Main Graph + Inspector Grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
        {/* Flow Canvas */}
        <div className="h-[540px] overflow-hidden rounded-xl border border-zinc-800 bg-black shadow-xl relative">
          <ReactFlow
            nodes={renderNodes}
            edges={renderEdges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.18 }}
            minZoom={0.2}
            maxZoom={1.6}
            nodesConnectable={false}
            elementsSelectable
            onNodeClick={(_, node) => setSelectedNode(node.id)}
            onNodeMouseEnter={(_, node) => setHoveredNode(node.id)}
            onNodeMouseLeave={() => setHoveredNode(null)}
            onPaneClick={() => setSelectedNode(null)}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={24} size={1} color="#27272a" />
            <MiniMap
              pannable
              zoomable
              nodeColor="#27272a"
              maskColor="rgba(0, 0, 0, 0.85)"
              className="!bg-zinc-950"
            />
            <Controls
              showInteractive={false}
              className="!border-zinc-800 !bg-zinc-900"
            />
          </ReactFlow>

          {/* Edge Stability Legend Bar */}
          <div className="pointer-events-none absolute top-3 right-3 flex flex-wrap items-center gap-2.5 rounded-lg border border-zinc-800 bg-zinc-950/90 px-3 py-1.5 text-[10px] font-mono text-zinc-300 shadow-md">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-white inline-block" /> Intervention
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-zinc-400 inline-block" /> Solid: High Stability (&ge;75%)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-zinc-600 inline-block" /> Gray: Stable (50–74%)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-zinc-800 inline-block" /> Dashed: Exploratory
            </span>
          </div>

          {/* Quick Helper Floating Tip */}
          <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-zinc-800 bg-zinc-950/90 px-3 py-1.5 text-[11px] font-mono text-zinc-500 shadow-md">
            Click any node to inspect causal parents, edge stability &amp; state space
          </div>
        </div>

        {/* Node Inspector Drawer */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-5 shadow-xl flex flex-col justify-between">
          <AnimatePresence mode="wait">
            {selectedNode && selectedNodeMeta ? (
              <motion.div
                key={selectedNode}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="space-y-4"
              >
                <div className="flex items-start justify-between border-b border-zinc-800 pb-3">
                  <div>
                    <span className="rounded bg-white text-black font-bold px-2 py-0.5 font-mono text-xs">
                      {selectedNode}
                    </span>
                    <h4 className="mt-2 font-display text-base font-bold text-white">
                      {selectedNodeMeta.name}
                    </h4>
                    <p className="font-mono text-xs text-zinc-400">{selectedNodeMeta.category}</p>
                  </div>
                  <button
                    onClick={() => setSelectedNode(null)}
                    className="rounded p-1 text-zinc-400 hover:text-white cursor-pointer"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>

                <p className="text-xs text-zinc-300 leading-relaxed">{selectedNodeMeta.desc}</p>

                {/* Causal Dependencies with Edge Stability */}
                <div className="space-y-2">
                  <p className="font-mono text-[11px] font-bold text-zinc-300 uppercase">
                    Causal Relationships &amp; Edge Stability:
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="rounded-lg border border-zinc-800 bg-black p-2.5">
                      <span className="text-[10px] text-zinc-500 block">Parents (In-Degree):</span>
                      <span className="font-bold text-white text-sm">{incomingEdges.length}</span>
                      <div className="mt-1 space-y-1 max-h-24 overflow-y-auto">
                        {incomingEdges.map((e) => (
                          <div key={e.source} className="text-[10px] text-zinc-400 truncate">
                            &larr; <strong className="text-white">{e.source}</strong> ({e.support_category || 'DAG'})
                          </div>
                        ))}
                        {incomingEdges.length === 0 && <span className="text-[10px] text-zinc-600">Root Node</span>}
                      </div>
                    </div>
                    <div className="rounded-lg border border-zinc-800 bg-black p-2.5">
                      <span className="text-[10px] text-zinc-500 block">Children (Out-Degree):</span>
                      <span className="font-bold text-white text-sm">{outgoingEdges.length}</span>
                      <div className="mt-1 space-y-1 max-h-24 overflow-y-auto">
                        {outgoingEdges.map((e) => (
                          <div key={e.target} className="text-[10px] text-zinc-400 truncate">
                            &rarr; <strong className="text-white">{e.target}</strong> ({e.support_category || 'DAG'})
                          </div>
                        ))}
                        {outgoingEdges.length === 0 && <span className="text-[10px] text-zinc-600">Terminal Node</span>}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Discretized Domain States */}
                <div>
                  <p className="font-mono text-[11px] font-bold text-zinc-300 uppercase mb-1.5">
                    CPT State Space ({selectedStates.length} states):
                  </p>
                  <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto p-1">
                    {selectedStates.map((state) => (
                      <span
                        key={state}
                        className="rounded border border-zinc-800 bg-black px-2 py-1 font-mono text-[11px] text-zinc-300"
                      >
                        {state}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-center text-zinc-500">
                <Info className="h-8 w-8 text-zinc-600 mb-2" />
                <p className="font-display text-sm font-bold text-white">Select a Graph Node</p>
                <p className="text-xs text-zinc-500 mt-1 max-w-[220px]">
                  Click any node in the Bayesian Network to view its parents, children, bootstrap stability, and state spaces.
                </p>
              </div>
            )}
          </AnimatePresence>

          <div className="mt-4 pt-3 border-t border-zinc-800 text-[11px] text-zinc-500 font-mono flex items-center justify-between">
            <span>23 Causal Edges</span>
            <span className="text-zinc-400">BDeu (ESS=10)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
