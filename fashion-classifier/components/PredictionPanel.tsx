import { CLASS_NAMES } from "@/lib/classes";

type Props = {
  probabilities: number[] | null;
};

export default function PredictionPanel({ probabilities }: Props) {
  const top =
    probabilities && probabilities.length
      ? probabilities.indexOf(Math.max(...probabilities))
      : -1;

  return (
    <div className="w-full max-w-sm mx-auto">
      <p className="mb-3 font-tag text-[10px] uppercase tracking-tagwide text-muted">
        Composition
      </p>
      <ul className="flex flex-col gap-2">
        {CLASS_NAMES.map((name, i) => {
          const p = probabilities?.[i] ?? 0;
          const isTop = i === top;
          return (
            <li key={name} className="flex items-center gap-3">
              <span
                className={`w-24 shrink-0 truncate font-tag text-[11px] uppercase tracking-tight ${
                  isTop ? "text-ink" : "text-muted"
                }`}
              >
                {name}
              </span>
              <span className="relative h-2.5 flex-1 overflow-hidden rounded-none bg-canvas/70">
                <span
                  className={`absolute inset-y-0 left-0 transition-all duration-500 ease-out ${
                    isTop ? "bg-thread" : "bg-denim/50"
                  }`}
                  style={{ width: `${Math.round(p * 100)}%` }}
                />
              </span>
              <span className="w-10 shrink-0 text-right font-tag text-[11px] text-muted">
                {Math.round(p * 100)}%
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
