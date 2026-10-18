import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ERROR_TITLES, ErrorBanner } from "@/components/ErrorBanner";
import { ApiError, ERROR_CODES, type ApiErrorCode } from "@/lib/api";

const ALL_CODES: ApiErrorCode[] = [
  ...ERROR_CODES,
  "network_error",
  "invalid_request",
  "http_error",
  "bad_response",
];

describe("ErrorBanner", () => {
  it("has a title for every server error code", () => {
    expect(ERROR_CODES.every((code) => ERROR_TITLES[code])).toBe(true);
  });

  it.each(ALL_CODES)("renders %s as an alert with title, code and message", (code) => {
    render(<ErrorBanner error={new ApiError({ code, message: `message for ${code}` })} />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveAttribute("data-code", code);
    expect(alert).toHaveTextContent(ERROR_TITLES[code]);
    expect(screen.getByText(code, { selector: "code" })).toBeInTheDocument();
    expect(screen.getByText(`message for ${code}`)).toBeInTheDocument();
  });

  it("prefers the server hint and shows status details", () => {
    render(
      <ErrorBanner
        context="Signal and Summary"
        error={
          new ApiError({
            code: "source_blocked",
            message: "403 from the publisher",
            hint: "Use the press release instead.",
            status: 502,
            retryable: true,
            upstreamStatus: 403,
          })
        }
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Signal and Summary: Blocked by the source");
    expect(screen.getByText("HTTP 502")).toBeInTheDocument();
    expect(screen.getByText("Use the press release instead.")).toBeInTheDocument();
    expect(screen.getByText(/Upstream status 403\. Retrying may help\./)).toBeInTheDocument();
  });

  it("falls back to a local hint when the server sent none", () => {
    render(<ErrorBanner error={{ code: "paywalled", message: "Subscriber-only article" }} />);
    expect(screen.getByText(/behind a paywall/i)).toBeInTheDocument();
  });
});
