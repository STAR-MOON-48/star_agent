import { Time, type System, type SystemContext } from "@star-world/core";

import { participantQuery } from "./components.js";

export class SceneMotionSystem implements System<typeof participantQuery> {
  readonly query = participantQuery.withResource(Time, "read");

  update(context: SystemContext<typeof this.query>): void {
    const delta = Math.min(context.resource(Time).delta, 0.2);
    const amount = 1 - Math.exp(-delta * 5);
    for (const row of context.rows()) {
      row.layout.x += (row.layout.targetX - row.layout.x) * amount;
      row.layout.y += (row.layout.targetY - row.layout.y) * amount;
    }
  }
}
