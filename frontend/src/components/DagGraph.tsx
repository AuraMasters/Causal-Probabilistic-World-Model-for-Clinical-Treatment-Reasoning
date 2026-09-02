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
  strat: 'Stratification' in { Stratification: 1 }
    ? { name: 'Antiretroviral Stratum', category: 'Treatment History', desc: 'Stratification category based on prior ART' }
    : { name: 'Antiretroviral Stratum', category: 'Treatment History', desc: 'Stratification category based on prior ART' },
  symptom: { name: 'Symptomatic Indicator', category: 'Clinical Status', desc: 'Presence of HIV symptoms at baseline' },
  cd40: { name: 'Baseline CD4 T-Cell Count', category: 'Biomarker', desc: 'Baseline helper T-cell count (cells/mm³)' },
  cd80: { name: 'Baseline CD8 T-Cell Count', category: 'Biomarker', desc: 'Baseline cytotoxic T-cell count (cells/mm³)' },
  trt: { name: 'Treatment Regimen (Intervention)', category: 'Intervention', desc: 'Randomized treatment arm (4 options)' },
  label: { name: 'Primary Trial Outcome (Label)', category: 'Outcome', desc: '50% CD4 decline, AIDS endpoint, or death' },
}

const NODE_ACCENTS: Record<string, string> = {
  trt: 'mint',
  label: 'rose',
  cd40: 'cyan',
  cd80: 'cyan',
  karnof: 'cyan',
  preanti: 'cyan',
  z30: 'cyan',
}

type DagNodeData = {
  label: string
  name: string
  category: string
  accent: string
  dimmed: boolean
  isTarget: boolean
  isSource: boolean
}

function DagNode({ data, selected }: NodeProps) {
  const { label, name, category, accent, dimmed, isTarget, isSource } = data as DagNodeData

  const borderClass =
    accent === 'mint'
      ? 'border-mint-400/60 shadow-mint-400/20'
      : accent === 'rose'
        ? 'border-rose-400/60 shadow-rose-400/20'
        : 'border-cyan-400/40 shadow-cyan-400/15'

  const badgeBg =
    accent === 'mint'
      ? 'bg-mint-400/20 text-mint-200'
      : accent === 'rose'
        ? 'bg-rose-400/20 text-rose-200'
        : 'bg-cyan-400/20 text-cyan-200'

  return (
    <div
      className={`relative w-[170px] rounded-2xl border bg-ink-900/95 p-3 shadow-xl backdrop-blur transition-all duration-200 ${borderClass} ${
        selected ? 'ring-2 ring-cyan-300 shadow-cyan-400/40 scale-105' : ''
      } ${dimmed ? 'opacity-30' : 'opacity-100'} ${
        isSource ? 'ring-2 ring-cyan-400 animate-pulse' : ''
      } ${isTarget ? 'ring-2 ring-mint-300' : ''}`}
    >
      <Handle type="target" position={Position.Left} className="!h-2 !w-2 !border-0 !bg-cyan-400" />
      <div className="flex items-center justify-between">
        <span className={`rounded-md px-1.5 py-0.5 font-mono text-[11px] font-extrabold ${badgeBg}`}>
          {label}
        </span>
        <span className="font-mono text-[9px] text-slate-400 uppercase">{category}</span>
      </div>
      <p className="mt-1.5 truncate font-display text-xs font-bold text-white">{name}</p>
      <Handle type="source" position={Position.Right} className="!h-2 !w-2 !border-0 !bg-mint-400" />
    </div>
  )
}

const nodeTypes: NodeTypes = { dagNode: DagNode }

function layoutDagre(nodes: FlowNode[], edges: FlowEdge[]) {
  const g = new dagre.graphlib.Graph()
  g.setGraph({
    rankdir: 'LR',
    nodesep: 45,
    ranksep: 90,
    marginx: 30,
    marginy: 30,
  })
  g.setDefaultEdgeLabel(() => ({}))

  nodes.forEach((node) => {
    g.setNode(node.id, { width: 170, height: 70 })
  })

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target)
  })

  dagre.layout(g)

  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = g.node(node.id)
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - 85,
        y: nodeWithPosition.y - 35,
      },
    }
  })

  return { nodes: layoutedNodes, edges }
}

interface DagGraphProps {
  overview: Overview
}

export function DagGraph({ overview }: DagGraphProps) {
  return (
    <ReactFlowProvider>
      <DagGraphInner overview={overview} />
    </ReactFlowProvider>
  )
}

function DagGraphInner({ overview }: DagGraphProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>('trt')
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [activeFilter, setActiveFilter] = useState<'all' | 'intervention' | 'biomarkers' | 'outcome'>('all')

  const reactFlow = useReactFlow()

  const { rawNodes, rawEdges } = useMemo(() => {
    const edges: FlowEdge[] = overview.dag.map((edge, index) => ({
      id: `edge-${index}`,
      source: edge.source,
      target: edge.target,
      animated: false,
      style: { stroke: '#1d3158', strokeWidth: 1.6 },
    }))

    const nodes: FlowNode[] = Object.keys(NODE_DESCRIPTIONS).map((variable) => {
      const meta = NODE_DESCRIPTIONS[variable]
      return {
        id: variable,
        type: 'dagNode',
        position: { x: 0, y: 0 },
        data: {
          label: variable,
          name: meta.name,
          category: meta.category,
          accent: NODE_ACCENTS[variable] ?? 'slate',
          dimmed: false,
          isTarget: false,
          isSource: false,
        },
      }
    })

    const layouted = layoutDagre(nodes, edges)
    return { rawNodes: layouted.nodes, rawEdges: layouted.edges }
  }, [overview])

  const highlight = hoveredNode ?? selectedNode

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
      } else if (activeFilter === 'outcome') {
        dimmed = node.id !== 'label'
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
      return {
        ...edge,
        animated: isHighlighted,
        style: {
          stroke: isHighlighted ? '#92eeff' : '#1d3158',
          strokeWidth: isHighlighted ? 2.8 : 1.6,
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
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-cyan-400/20 bg-ink-900/90 p-3.5 shadow-md">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="font-mono text-xs font-bold text-slate-400 uppercase mr-2 flex items-center gap-1.5">
            <GitBranch className="h-4 w-4 text-cyan-300" /> Focus View:
          </span>
          <button
            onClick={() => {
              setActiveFilter('all')
              setSelectedNode(null)
            }}
            className={`rounded-lg px-3 py-1 text-xs font-semibold transition-all cursor-pointer ${
              activeFilter === 'all' && !selectedNode
                ? 'bg-cyan-400/20 text-cyan-200 border border-cyan-400/40 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            All 17 Nodes
          </button>
          <button
            onClick={handleFocusTrtPath}
            className={`rounded-lg px-3 py-1 text-xs font-semibold transition-all cursor-pointer ${
              selectedNode === 'trt'
                ? 'bg-mint-300/20 text-mint-200 border border-mint-300/40 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Treatment Pathway (trt &rarr; label)
          </button>
          <button
            onClick={() => {
              setActiveFilter('biomarkers')
              setSelectedNode('cd40')
            }}
            className={`rounded-lg px-3 py-1 text-xs font-semibold transition-all cursor-pointer ${
              activeFilter === 'biomarkers'
                ? 'bg-cyan-400/20 text-cyan-200 border border-cyan-400/40 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Key Biomarkers
          </button>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomIn}
            className="rounded-lg border border-slate-700 bg-ink-850 p-1.5 text-slate-300 hover:text-white hover:border-cyan-400 cursor-pointer"
            title="Zoom In"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleZoomOut}
            className="rounded-lg border border-slate-700 bg-ink-850 p-1.5 text-slate-300 hover:text-white hover:border-cyan-400 cursor-pointer"
            title="Zoom Out"
          >
            <Minimize2 className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleFitView}
            className="rounded-lg border border-slate-700 bg-ink-850 p-1.5 text-slate-300 hover:text-white hover:border-cyan-400 cursor-pointer"
            title="Fit View"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Main Graph + Inspector Grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_300px]">
        {/* Flow Canvas */}
        <div className="h-[540px] overflow-hidden rounded-2xl border border-cyan-400/25 bg-ink-900/90 shadow-xl relative">
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
            <Background gap={24} size={1} color="#0e1a30" />
            <MiniMap
              pannable
              zoomable
              nodeColor="#142340"
              maskColor="rgba(5, 10, 20, 0.75)"
              className="!bg-ink-900/90"
            />
            <Controls
              showInteractive={false}
              className="!border-cyan-400/25 !bg-ink-850"
            />
          </ReactFlow>

          {/* Quick Helper Floating Tip */}
          <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg border border-slate-700/60 bg-ink-950/90 px-3 py-1.5 text-[11px] font-mono text-slate-400 shadow-md">
            Click any node to inspect causal parents &amp; conditional distribution
          </div>
        </div>

        {/* Node Inspector Drawer */}
        <div className="rounded-2xl border border-cyan-400/25 bg-ink-900/95 p-5 shadow-xl flex flex-col justify-between">
          <AnimatePresence mode="wait">
            {selectedNode && selectedNodeMeta ? (
              <motion.div
                key={selectedNode}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="space-y-4"
              >
                <div className="flex items-start justify-between border-b border-slate-700/70 pb-3">
                  <div>
                    <span className="rounded-md bg-cyan-400/20 border border-cyan-400/35 px-2 py-0.5 font-mono text-xs font-bold text-mint-200">
                      {selectedNode}
                    </span>
                    <h4 className="mt-2 font-display text-base font-bold text-white">
                      {selectedNodeMeta.name}
                    </h4>
                    <p className="font-mono text-xs text-slate-400">{selectedNodeMeta.category}</p>
                  </div>
                  <button
                    onClick={() => setSelectedNode(null)}
                    className="rounded-lg p-1 text-slate-400 hover:text-white cursor-pointer"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed">{selectedNodeMeta.desc}</p>

                {/* Causal Dependencies */}
                <div className="space-y-2">
                  <p className="font-mono text-[11px] font-bold text-cyan-300 uppercase">
                    Causal Relationships:
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="rounded-xl border border-slate-700/60 bg-ink-950 p-2.5">
                      <span className="text-[10px] text-slate-400 block">Parents (In-Degree):</span>
                      <span className="font-bold text-cyan-200 text-sm">{incomingEdges.length}</span>
                      <div className="mt-1 text-[10px] text-slate-400 truncate">
                        {incomingEdges.map((e) => e.source).join(', ') || 'Root node'}
                      </div>
                    </div>
                    <div className="rounded-xl border border-slate-700/60 bg-ink-950 p-2.5">
                      <span className="text-[10px] text-slate-400 block">Children (Out-Degree):</span>
                      <span className="font-bold text-mint-200 text-sm">{outgoingEdges.length}</span>
                      <div className="mt-1 text-[10px] text-slate-400 truncate">
                        {outgoingEdges.map((e) => e.target).join(', ') || 'Terminal node'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Discretized Domain States */}
                <div>
                  <p className="font-mono text-[11px] font-bold text-cyan-300 uppercase mb-1.5">
                    CPT State Space ({selectedStates.length} states):
                  </p>
                  <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto p-1">
                    {selectedStates.map((state) => (
                      <span
                        key={state}
                        className="rounded-lg border border-slate-700/80 bg-ink-950 px-2 py-1 font-mono text-[11px] text-slate-200"
                      >
                        {state}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-center text-slate-400">
                <Info className="h-8 w-8 text-cyan-400/60 mb-2" />
                <p className="font-display text-sm font-bold text-white">Select a Graph Node</p>
                <p className="text-xs text-slate-400 mt-1 max-w-[200px]">
                  Click any node in the Bayesian Network to view its parents, children, and state spaces.
                </p>
              </div>
            )}
          </AnimatePresence>

          <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500 font-mono">
            Final Model: 23 Edges &middot; BDeu Prior
          </div>
        </div>
      </div>
    </div>
  )
}
