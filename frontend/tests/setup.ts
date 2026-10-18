import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// RTL only auto-cleans when test globals are enabled; this suite imports them explicitly.
afterEach(() => cleanup());
