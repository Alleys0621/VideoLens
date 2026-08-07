export interface Persona {
  id: string;
  name: string;
  tagline: string;
  avatar: string;
  accent: string;
}

export const PERSONAS: Persona[] = [
  { id: "alleys", name: "小艾", tagline: "温柔知心", avatar: "/personas/alleys.png", accent: "#6366F1" },
  { id: "du", name: "阿毒", tagline: "毒舌损友", avatar: "/personas/du.png", accent: "#10B981" },
  { id: "lao_ju", name: "老剧", tagline: "高冷剧评人", avatar: "/personas/lao_ju.png", accent: "#0EA5E9" },
];

export const DEFAULT_PERSONA_ID = "alleys";

export function getPersona(id?: string | null): Persona {
  return PERSONAS.find((p) => p.id === id) ?? PERSONAS[0];
}
