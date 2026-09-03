type Props = {
  label: string | null;
  confidence: number | null; // 0..1
};

export default function GarmentTag({ label, confidence }: Props) {
  return (
    <div className="relative mx-auto w-full max-w-xs">
      <div className="stitched flex items-center gap-4 bg-paper px-5 py-4 shadow-swatch">
        <span className="grommet shrink-0" aria-hidden />
        <div className="min-w-0">
          <p className="font-tag text-[10px] uppercase tracking-tagwide text-muted">
            Reads as
          </p>
          <p className="truncate font-display text-2xl font-medium text-ink">
            {label ?? "—"}
          </p>
          {confidence !== null && (
            <p className="font-tag text-xs tracking-tagwide text-thread">
              {(confidence * 100).toFixed(1)}% confidence
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
