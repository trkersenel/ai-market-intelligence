"use client";

/**
 * The interactive ecosystem map.
 *
 * A force simulation is the right choice *here*, unlike for the supply chain:
 * this view has no inherent direction to preserve. The question it answers is
 * "what is clustered around what", and springs answer that well — companies
 * with many shared relationships end up near each other without anyone
 * deciding they should.
 *
 * The simulation runs once, to completion, before the first paint rather than
 * animating into place. A graph that visibly settles looks impressive for two
 * seconds and is unusable for those two seconds, and a reader who clicks a
 * moving node misses. Layout is computed synchronously in a worker-free tick
 * loop — at this size it takes a few milliseconds.
 */

import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import type { EcosystemGraph, EntityNode, RelationshipEdge } from "@/lib/api/types";
import { cn } from "@/lib/cn";

const WIDTH = 900;
const HEIGHT = 520;

/** Colour by layer of the stack, so the map reads as an industry rather than a
 *  set of dots. Falls back to muted for anything untagged. */
const TAG_COLOURS: [string, string][] = [
  ["foundry", "oklch(0.72 0.15 162)"],
  ["hbm", "oklch(0.7 0.16 258)"],
  ["memory", "oklch(0.7 0.16 258)"],
  ["gpu", "oklch(0.75 0.16 85)"],
  ["ai-compute", "oklch(0.75 0.16 85)"],
  ["semicap", "oklch(0.68 0.15 320)"],
  ["cloud", "oklch(0.7 0.14 220)"],
  ["ai-lab", "oklch(0.72 0.17 20)"],
  ["power", "oklch(0.74 0.13 55)"],
  ["cooling", "oklch(0.74 0.13 55)"],
  ["networking", "oklch(0.68 0.12 195)"],
];

function colourFor(node: EntityNode): string {
  for (const [tag, colour] of TAG_COLOURS) {
    if (node.tags.includes(tag)) return colour;
  }
  return "var(--color-muted)";
}

interface PositionedNode extends SimulationNodeDatum {
  id: string;
  node: EntityNode;
  radius: number;
  /** Half the rendered label width, in user units. Drives collision, because a
   *  circle can clear its neighbour comfortably while "CoWoS advanced
   *  packaging" still lands on top of "KLAC". */
  labelHalfWidth: number;
}

type PositionedLink = SimulationLinkDatum<PositionedNode> & { edge: RelationshipEdge };

/** Run the simulation to rest and return final coordinates. */
function layout(graph: EcosystemGraph): { nodes: PositionedNode[]; links: PositionedLink[] } {
  const degree = new Map<string, number>();
  for (const edge of graph.edges) {
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
  }

  const nodes: PositionedNode[] = graph.nodes.map((node) => {
    const label = node.symbol ?? node.name;
    return {
      id: node.slug,
      node,
      // Size by connectedness: the structurally important companies should be
      // visually important. Square-rooted so a node with twenty edges is not
      // twenty times the area of one with a single edge.
      radius: node.slug === graph.root ? 26 : 8 + Math.sqrt(degree.get(node.slug) ?? 1) * 3.5,
      // ~3.1px per character at 11px, which is close enough for a collision
      // budget and far cheaper than measuring text in the DOM.
      labelHalfWidth: (label.length * 3.1) / 2,
    };
  });

  const byId = new Map(nodes.map((node) => [node.id, node]));
  const links: PositionedLink[] = graph.edges
    .filter((edge) => byId.has(edge.source) && byId.has(edge.target))
    .map((edge) => ({ source: edge.source, target: edge.target, edge }));

  const simulation = forceSimulation(nodes)
    .force("link", forceLink<PositionedNode, PositionedLink>(links).id((d) => d.id).distance(135))
    .force("charge", forceManyBody().strength(-420))
    .force("center", forceCenter(WIDTH / 2, HEIGHT / 2))
    // Collision covers the label as well as the circle. Without the label term
    // the nodes cleared each other while their captions overlapped -- "CoWoS
    // advanced packaging" sat on top of "KLAC".
    .force(
      "collide",
      forceCollide<PositionedNode>()
        .radius((d) => Math.max(d.radius + 18, d.labelHalfWidth + 10))
        .strength(0.9),
    )
    .stop();

  // Enough ticks to converge at this size. Running to completion here rather
  // than animating means the first paint is the final layout.
  simulation.tick(320);
  return { nodes, links };
}

export function EcosystemMap({ graph }: { graph: EcosystemGraph }) {
  const router = useRouter();
  const [hovered, setHovered] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const { nodes, links } = useMemo(() => layout(graph), [graph]);

  const connected = useMemo(() => {
    if (!hovered) return null;
    const set = new Set<string>([hovered]);
    for (const link of links) {
      const source = (link.source as PositionedNode).id;
      const target = (link.target as PositionedNode).id;
      if (source === hovered) set.add(target);
      if (target === hovered) set.add(source);
    }
    return set;
  }, [hovered, links]);

  const dimmed = (id: string) => connected !== null && !connected.has(id);

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="h-[520px] w-full touch-none select-none"
      role="img"
      aria-label={`Ecosystem map centred on ${graph.root}, ${nodes.length} entities and ${links.length} relationships`}
    >
      <g>
        {links.map((link, index) => {
          const source = link.source as PositionedNode;
          const target = link.target as PositionedNode;
          const faded = dimmed(source.id) || dimmed(target.id);
          return (
            <line
              key={index}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke={
                link.edge.kind === "competes_with"
                  ? "var(--color-down)"
                  : "var(--color-line-strong)"
              }
              // Width encodes materiality, opacity encodes confidence. Two
              // different questions, so two different visual channels rather
              // than one blended score that answers neither.
              strokeWidth={0.6 + link.edge.weight * 2.2}
              strokeOpacity={faded ? 0.06 : 0.2 + link.edge.confidence * 0.5}
              strokeDasharray={link.edge.evidence === "inferred" ? "4 3" : undefined}
            />
          );
        })}
      </g>

      <g>
        {nodes.map((node) => {
          const isRoot = node.id === graph.root;
          const faded = dimmed(node.id);
          return (
            <g
              key={node.id}
              transform={`translate(${node.x},${node.y})`}
              className="cursor-pointer"
              opacity={faded ? 0.18 : 1}
              onMouseEnter={() => setHovered(node.id)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => router.push(`/ecosystem/${node.id}`)}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter") router.push(`/ecosystem/${node.id}`);
              }}
              aria-label={node.node.name}
            >
              <circle
                r={node.radius}
                fill={colourFor(node.node)}
                fillOpacity={isRoot ? 0.95 : 0.75}
                stroke={isRoot ? "var(--color-ink)" : "var(--color-canvas)"}
                strokeWidth={isRoot ? 2.5 : 1.5}
              />
              <text
                y={node.radius + 13}
                textAnchor="middle"
                className={cn(
                  "pointer-events-none fill-[var(--color-ink)]",
                  isRoot ? "text-[13px] font-semibold" : "text-[11px]",
                )}
                style={{ paintOrder: "stroke", stroke: "var(--color-canvas)", strokeWidth: 3 }}
              >
                {node.node.symbol ?? node.node.name}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

/** Legend for the colour and line encodings, so the map is self-explaining. */
export function MapLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-[var(--color-line)] px-4 py-2.5 text-[10px] text-[var(--color-subtle)]">
      {[
        ["Foundry", "oklch(0.72 0.15 162)"],
        ["Memory / HBM", "oklch(0.7 0.16 258)"],
        ["Accelerators", "oklch(0.75 0.16 85)"],
        ["Equipment", "oklch(0.68 0.15 320)"],
        ["Cloud", "oklch(0.7 0.14 220)"],
        ["AI labs", "oklch(0.72 0.17 20)"],
        ["Power / cooling", "oklch(0.74 0.13 55)"],
      ].map(([label, colour]) => (
        <span key={label} className="inline-flex items-center gap-1.5">
          <span aria-hidden className="size-2 rounded-full" style={{ backgroundColor: colour }} />
          {label}
        </span>
      ))}
      <span className="ml-auto inline-flex items-center gap-3">
        <span>Thickness = materiality</span>
        <span>Opacity = confidence</span>
        <span className="inline-flex items-center gap-1">
          <svg width="18" height="4" aria-hidden>
            <line
              x1="0"
              y1="2"
              x2="18"
              y2="2"
              stroke="var(--color-line-strong)"
              strokeWidth="1.5"
              strokeDasharray="4 3"
            />
          </svg>
          inferred
        </span>
      </span>
    </div>
  );
}
