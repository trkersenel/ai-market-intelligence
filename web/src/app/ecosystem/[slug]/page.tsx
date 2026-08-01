import type { Metadata } from "next";
import { EcosystemView } from "@/components/graph/ecosystem-view";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  return { title: `${slug} ecosystem · Market Intelligence` };
}

export default async function EcosystemPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  return <EcosystemView identifier={slug} />;
}
