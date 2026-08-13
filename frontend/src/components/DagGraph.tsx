import { useMemo, useState } from 'react'
import dagre from 'dagre'
import { AnimatePresence, motion } from 'framer-motion'
import { X } from 'lucide-react'
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  type Edge as FlowEdge,
  type Node as FlowNode,
  type NodeProps,
  type NodeTypes,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import type { Overview } from '../lib/types'

const NODE_DESCRIPTIONS: Record<string, string> = {
  age: 'Age',
  wtkg: 'Weight (kg)',
  hemo: 'Hemophilia',
  homo: 'Homosexuality',
  drugs: 'IV drug use',
  karnof: 'Karnofsky score',
  oprior: 'Prior opportunistic infection',
  z30: 'Prior zidovudine use',
  preanti: 'Pre-ART exposure',
  race: 'Race',
  gender: 'Gender',
  strat: 'Stratification',
  symptom: 'Symptomatic disease',
  cd40: 'CD4 count',
  cd80: 'CD8 count',
  trt: 'Treatment',
  label: 'Outcome',
}

const NODE_ACCENTS: Record<string, string> = {
  trt: 'mint',
  label: 'violet',
  cd40: 'cyan',
  cd80: 'cyan',
  z30: 'cyan',
}

type DagNodeData = {
  label: string
  description: string
  accent: string
  dimmed: boolean
}

function DagNode({ data, selected }: NodeProps) {
  const { label, description, accent, dimmed } = data as DagNodeData

  const borderClass =
    accent === 'mint'
      ? 'border-mint-400/40'
      : accent === 'violet'
        ? 'border-violet-400/40'
        : 'border-cyan-400/30'
  const textClass =
    accent === 'mint'
      ? 'text-mint-300'
      : accent === 'violet'
        ? 'text-violet-300'
        : 'text-cyan-300'

  return (
    <div
      className={`relative w-[160px] rounded-xl border bg-ink-800/95 px-3 py-2.5 shadow-lg shadow-black/30 backdrop-blur transition-all duration-200 ${
        borderClass
      } ${selected ? 'ring-2 ring-cyan-400/50' : ''} ${dimmed ? 'opacity-40' : ''}`}
    >
      <Handle type="target" position={Position.Left} className="!h-1.5 !w-1.5 !border-0 !bg-slate-500" />
      <p className={`font-mono text-[13px] font-bold tracking-wide ${textClass}`}>{label}</p>
      <p className="mt-0.5 truncate text-[10px] text-slate-400">{description}</p>
      <Handle type="source" position={Position.Right} className="!h-1.5 !w-1.5 !border-0 !bg-slate-500" />
    </div>
  )
}

const nodeTypes: NodeTypes = { dagNode: DagNode }

interface DagGraphProps {
  overview: Overview
}

export function DagGraph({ overview }: DagGraphProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  const { nodes, edges } = useMemo(() => {
    const rawEdges: FlowEdge[] = overview.dag.map((edge, index) => ({
      id: `edge-${index}`,
      source: edge.source,
      target: edge.target,
      animated: false,
      style: { stroke: '#3b4a6b', strokeWidth: 1.5 },
    }))

    const rawNodes: FlowNode[] = Object.keys(NODE_DESCRIPTIONS).map((variable) => ({
      id: variable,
      type: 'dagNode',
      position: { x: 0, y: 0 },
      data: {
        label: variable,
        description: NODE_DESCRIPTIONS[variable],
        accent: NODE_ACCENTS[variable] ?? 'slate',
        dimmed: false,
      },
    }))

    const layouted = layoutDagre(rawNodes, rawEdges)
    return { nodes: layouted.nodes, edges: layouted.edges }
  }, [overview])

  const highlight = hoveredNode ?? selectedNode

  const renderNodes = useMemo(
    () =>
      nodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          dimmed: highlight !== null && highlight !== node.id && !edges.some(
            (edge) => edge.source === highlight && edge.target === node.id,
          ) && !edges.some(
            (edge) => edge.target === highlight && edge.source === node.id,
          ),
        },
      })),
    [nodes, edges, highlight],
  )

  const renderEdges = useMemo(
    () =>
      edges.map((edge) => ({
        ...edge,
        animated: highlight !== null && (edge.source === highlight || edge.target === highlight),
        style: {
          stroke: highlight !== null && (edge.source === highlight || edge.target === highlight)
            ? '#34d9b0'
            : '#3b4a6b',
          strokeWidth: highlight !== null && (edge.source === highlight || edge.target === highlight) ? 2.4 : 1.5,
        },
      })),
    [edges, highlight],
  )

  const selectedStates = selectedNode ? overview.states[selectedNode] ?? [] : []

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_260px]">
      <div className="h-[520px] overflow-hidden rounded-2xl border border-slate-500/20 bg-ink-900/60">
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
          <Background gap={22} size={1} color="#1c2847" />
          <MiniMap
            pannable
            zoomable
            nodeColor="#1a2540"
            maskColor="rgba(6, 10, 20, 0.75)"
            className="!bg-ink-900/80"
          />
          <Controls className="!border-slate-600/40 !bg-ink-800 !fill-slate-300" />
        </ReactFlow>
      </div>

      <AnimatePresence mode="wait">
        {selectedNode ? (
          <motion.aside
            key={selectedNode}
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 12 }}
            className="card-surface h-fit rounded-2xl p-5"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="font-mono text-xs tracking-wide text-cyan-300 uppercase">Selected node</p>
                <h3 className="mt-1 font-display text-lg font-semibold text-slate-50">{selectedNode}</h3>
                <p className="mt-0.5 text-sm text-slate-400">{NODE_DESCRIPTIONS[selectedNode]}</p>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="rounded-lg p-1 text-slate-400 transition-colors hover:bg-ink-700 hover:text-slate-200"
                aria-label="Close node details"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-4">
              <p className="mb-2 font-mono text-[11px] font-medium tracking-[0.18em] text-slate-500 uppercase">
                Model states
              </p>
              <div className="flex flex-wrap gap-1.5">
                {selectedStates.map((state) => (
                  <span
                    key={state}
                    className="rounded-md border border-slate-600/40 bg-ink-800 px-2 py-1 font-mono text-[11px] text-slate-300"
                  >
                    {state}
                  </span>
                ))}
              </div>
            </div>
            <div className="mt-4 border-t border-slate-600/20 pt-3">
              <p className="text-[11px] leading-relaxed text-slate-500">
                {selectedNode === 'label'
                  ? '0 = desirable study outcome · 1 = undesirable study outcome'
                  : `Connections from this node are highlighted. Drag nodes to explore the 23-edge DAG.`}
              </p>
            </div>
          </motion.aside>
        ) : (
          <motion.aside
            key="hint"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="card-surface h-fit rounded-2xl p-5"
          >
            <p className="font-mono text-[11px] font-medium tracking-[0.18em] text-slate-500 uppercase">Interactions</p>
            <ul className="mt-3 space-y-2.5 text-sm text-slate-300">
              <li>Click a node to inspect its model states.</li>
              <li>Hover a node to highlight its edges.</li>
              <li>Drag to rearrange · scroll to zoom.</li>
            </ul>
            <div className="mt-4 rounded-lg border border-slate-600/20 bg-ink-800/70 p-3">
              <p className="font-mono text-[11px] text-slate-400">
                <span className="text-mint-300">trt → label</span> is the causal treatment edge.
              </p>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  )
}

function layoutDagre(nodes: FlowNode[], edges: FlowEdge[]) {
  const graph = new dagre.graphlib.Graph()
  graph.setDefaultEdgeLabel(() => ({}))
  graph.setGraph({ rankdir: 'LR', nodesep: 28, ranksep: 90, marginx: 10, marginy: 10 })

  nodes.forEach((node) => {
    graph.setNode(node.id, { width: 164, height: 58 })
  })
  edges.forEach((edge) => {
    graph.setEdge(edge.source, edge.target)
  })

  dagre.layout(graph)

  const layoutedNodes = nodes.map((node) => {
    const position = graph.node(node.id)
    return {
      ...node,
      position: {
        x: position.x - 164 / 2,
        y: position.y - 58 / 2,
      },
    }
  })

  return { nodes: layoutedNodes, edges }
}
