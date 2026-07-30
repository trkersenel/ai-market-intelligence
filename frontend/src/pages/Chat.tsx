/**
 * Grounded question answering.
 *
 * The interface is built around the platform's central promise: an answer is only
 * as good as its sources. So citations are not a footnote -- they sit beside the
 * answer, numbered to match the inline markers, and a refusal is rendered as a
 * first-class outcome rather than an error.
 *
 * Confidence is displayed with what produced it, and labelled as retrieval
 * strength rather than correctness. Those are different claims, and a number
 * shown without that distinction invites reading the second when the API means
 * the first.
 */

import { useMutation } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { ChatAnswer } from "../lib/types";
import { formatDate } from "../lib/format";

const SUGGESTIONS = [
  "What is happening with memory chip prices and HBM demand?",
  "Why did SK Hynix stock move?",
  "Which companies are exposed to advanced packaging?",
  "Summarise this week's semiconductor news.",
] as const;

interface Turn {
  question: string;
  answer?: ChatAnswer;
  error?: unknown;
}

export function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const conversationId = useRef<string | undefined>(undefined);
  const endRef = useRef<HTMLDivElement>(null);

  const ask = useMutation({
    mutationFn: (question: string) => api.chat.ask(question, conversationId.current),
    onSuccess: (answer) => {
      conversationId.current = answer.conversation_id;
      setTurns((current) => {
        const next = [...current];
        next[next.length - 1] = { question: answer.question, answer };
        return next;
      });
      requestAnimationFrame(() => endRef.current?.scrollIntoView({ behavior: "smooth" }));
    },
    onError: (error) => {
      setTurns((current) => {
        const next = [...current];
        const last = next[next.length - 1];
        if (last) next[next.length - 1] = { ...last, error };
        return next;
      });
    },
  });

  const submit = (question: string) => {
    const trimmed = question.trim();
    if (!trimmed || ask.isPending) return;
    setTurns((current) => [...current, { question: trimmed }]);
    setDraft("");
    ask.mutate(trimmed);
  };

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      {turns.length === 0 && (
        <section className="card">
          <h1 className="text-base font-semibold tracking-tight">Ask about the market</h1>
          <p className="mt-1 text-xs leading-relaxed text-ink-secondary">
            Every answer is built from retrieved documents and cites them. When the
            corpus does not cover a question, the assistant says so rather than
            guessing.
          </p>
          <ul className="mt-3 space-y-1.5">
            {SUGGESTIONS.map((suggestion) => (
              <li key={suggestion}>
                <button
                  type="button"
                  onClick={() => {
                    submit(suggestion);
                  }}
                  className="focusable w-full rounded border border-hairline px-2.5 py-2 text-left text-xs text-ink-secondary transition-colors hover:border-hairline-strong hover:text-ink"
                >
                  {suggestion}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {turns.map((turn, index) => (
        <TurnView key={index} turn={turn} pending={ask.isPending && index === turns.length - 1} />
      ))}

      <div ref={endRef} />

      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit(draft);
        }}
        className="sticky bottom-4 flex gap-2 rounded-lg border border-hairline bg-surface p-2"
      >
        <label className="flex-1">
          <span className="sr-only">Your question</span>
          {/* Explicit type, so implicit form submission on Enter is
              unambiguous. The form's onSubmit stays the single submit path --
              adding a keydown handler as well would double-submit in browsers
              where implicit submission already works. */}
          <input
            type="text"
            value={draft}
            onChange={(event) => {
              setDraft(event.target.value);
            }}
            placeholder="Ask about a company, a move, or the sector…"
            className="focusable w-full rounded bg-transparent px-2 py-1.5 text-sm text-ink placeholder:text-ink-muted"
          />
        </label>
        <button
          type="submit"
          disabled={ask.isPending || draft.trim().length === 0}
          className="focusable rounded bg-series-1 px-3 py-1.5 text-xs font-medium text-white transition-opacity disabled:opacity-40"
        >
          {ask.isPending ? "Asking…" : "Ask"}
        </button>
      </form>
    </div>
  );
}

function TurnView({ turn, pending }: { turn: Turn; pending: boolean }) {
  return (
    <section className="space-y-2">
      <p className="ml-auto max-w-[85%] rounded-lg bg-surface-raised px-3 py-2 text-sm text-ink">
        {turn.question}
      </p>

      {pending && (
        <div className="card">
          <p className="text-xs text-ink-muted">Retrieving sources…</p>
        </div>
      )}

      {turn.error !== undefined && (
        <div className="card" role="alert">
          {/* `unknown` is the honest type for a thrown value, so it is
              narrowed here rather than cast: anything that is not our own
              ApiError gets a generic message instead of being rendered raw. */}
          <p className="text-sm text-ink">
            {turn.error instanceof ApiError
              ? turn.error.message
              : "The question could not be answered."}
          </p>
          {turn.error instanceof ApiError && turn.error.requestId && (
            <p className="mt-1 text-2xs tabular text-ink-muted">
              request {turn.error.requestId}
            </p>
          )}
        </div>
      )}

      {turn.answer && <AnswerView answer={turn.answer} />}
    </section>
  );
}

function AnswerView({ answer }: { answer: ChatAnswer }) {
  return (
    <div className="card">
      {answer.refused ? (
        // A refusal is an outcome, not a failure: the platform is designed to
        // decline rather than fabricate, so it is styled as information.
        <div className="flex gap-2">
          <span aria-hidden className="mt-1 h-2 w-2 shrink-0 rounded-full bg-status-warning" />
          <div>
            <p className="text-2xs uppercase tracking-wide text-ink-muted">
              Not answerable from the corpus
            </p>
            <p className="mt-1 text-sm leading-relaxed text-ink">{answer.answer}</p>
          </div>
        </div>
      ) : (
        <p className="whitespace-pre-line text-sm leading-relaxed text-ink">{answer.answer}</p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-hairline pt-2 text-2xs text-ink-muted">
        {!answer.refused && (
          <span className="tabular">
            retrieval confidence {Math.round(answer.confidence * 100)}%
          </span>
        )}
        <span className="tabular">{answer.retrieved} passages retrieved</span>
        <span>{answer.model_name}</span>
        {/* Stated plainly: an extractive answer quotes its sources rather than
            synthesising them, and reads very differently. */}
        {answer.extractive && (
          <span className="rounded border border-hairline px-1 py-px">
            extractive — quoted, not synthesised
          </span>
        )}
      </div>

      {answer.citations.length > 0 && (
        <ol className="mt-2 space-y-1.5">
          {answer.citations.map((citation) => (
            <li key={citation.number} className="flex gap-2 text-2xs">
              <span className="tabular text-ink-muted">[{citation.number}]</span>
              <div className="min-w-0">
                {citation.url ? (
                  <a
                    href={citation.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="focusable rounded text-ink-secondary hover:text-ink hover:underline"
                  >
                    {citation.title ?? citation.url}
                  </a>
                ) : (
                  <span className="text-ink-secondary">{citation.title ?? "Untitled"}</span>
                )}
                <span className="ml-1.5 text-ink-muted">
                  {citation.published_at && formatDate(citation.published_at)}
                  {/* Which retriever found it: keyword, vector, or both. "Both"
                      is the strongest signal the passage is genuinely relevant. */}
                  {citation.matched_by.length > 0 && ` · ${citation.matched_by.join(" + ")}`}
                </span>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
