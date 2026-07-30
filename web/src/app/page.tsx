import type { Metadata } from "next";
import { Dashboard } from "@/components/dashboard/dashboard";

export const metadata: Metadata = {
  title: "Dashboard · Market Intelligence",
};

export default function HomePage() {
  return <Dashboard />;
}
