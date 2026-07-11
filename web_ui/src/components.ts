import { defineComponent, defineQuery, read, write } from "@star-world/core";

import type { ParticipantRole } from "./types.js";

export interface IdentityData {
  id: string;
  label: string;
  role: ParticipantRole;
}

export interface PresenceData {
  online: boolean;
  state: string;
  updatedAt: number;
}

export interface ActivityData {
  label: string;
  count: number;
}

export interface LayoutData {
  x: number;
  y: number;
  targetX: number;
  targetY: number;
}

export const Identity = defineComponent<IdentityData>("AgentLingWeb.Identity");
export const Presence = defineComponent<PresenceData>("AgentLingWeb.Presence");
export const Activity = defineComponent<ActivityData>("AgentLingWeb.Activity");
export const Layout = defineComponent<LayoutData>("AgentLingWeb.Layout");

export const participantQuery = defineQuery({
  identity: read(Identity),
  presence: write(Presence),
  activity: write(Activity),
  layout: write(Layout),
});
