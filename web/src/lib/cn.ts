import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge class names, letting later Tailwind utilities win over earlier ones.
 *
 * Without the merge step a component's default `px-3` and a caller's `px-6`
 * both land in the class list and the winner is whichever CSS rule was emitted
 * last -- which is a build-order detail, not a decision.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
