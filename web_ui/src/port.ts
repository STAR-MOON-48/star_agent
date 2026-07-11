import type { Server } from "node:http";
import type { AddressInfo } from "node:net";

export interface PortBindingOptions {
  host: string;
  preferredPort: number;
  maxAttempts?: number;
}

export async function listenOnAvailablePort(
  server: Server,
  options: PortBindingOptions,
): Promise<number> {
  const maxAttempts = Math.max(1, options.maxAttempts ?? 100);
  const lastPort = Math.min(65_535, options.preferredPort + maxAttempts - 1);

  for (let port = options.preferredPort; port <= lastPort; port += 1) {
    try {
      await listen(server, options.host, port);
      const address = server.address() as AddressInfo | null;
      if (!address) throw new Error("Web server did not expose a bound address");
      return address.port;
    } catch (error) {
      if (isAddressInUse(error) && port < lastPort) continue;
      throw error;
    }
  }

  throw new Error(
    `No available port found in ${options.preferredPort}-${lastPort}`,
  );
}

function listen(server: Server, host: string, port: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const onError = (error: Error): void => {
      server.off("listening", onListening);
      reject(error);
    };
    const onListening = (): void => {
      server.off("error", onError);
      resolve();
    };
    server.once("error", onError);
    server.once("listening", onListening);
    server.listen(port, host);
  });
}

function isAddressInUse(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as NodeJS.ErrnoException).code === "EADDRINUSE"
  );
}
