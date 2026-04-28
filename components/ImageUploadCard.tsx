"use client";

import { useMemo, useRef, useState } from "react";

interface ImageUploadCardProps {
  title: string;
  hint: string;
  imageUrl: string | null;
  onFileSelected: (file: File) => void;
  imageRef: React.RefObject<HTMLImageElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}

export default function ImageUploadCard({
  title,
  hint,
  imageUrl,
  onFileSelected,
  imageRef,
  canvasRef
}: ImageUploadCardProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const containerClassName = useMemo(() => {
    const baseClass =
      "panel relative overflow-hidden border-2 border-dashed transition-colors";
    return isDragging
      ? `${baseClass} border-cyan-400`
      : `${baseClass} border-slate-700 hover:border-slate-500`;
  }, [isDragging]);

  const acceptFile = (file?: File) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) return;
    onFileSelected(file);
  };

  return (
    <section
      className={containerClassName}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        const dropped = event.dataTransfer.files?.[0];
        acceptFile(dropped);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => {
          acceptFile(event.target.files?.[0]);
        }}
      />

      {!imageUrl ? (
        <button
          type="button"
          className="flex min-h-[300px] w-full flex-col items-center justify-center gap-3 px-6 py-10 text-center"
          onClick={() => inputRef.current?.click()}
        >
          <span className="text-base font-semibold text-cyan-300">{title}</span>
          <span className="text-sm text-slate-400">{hint}</span>
          <span className="rounded-md border border-slate-700 px-4 py-2 text-sm text-slate-200">
            点击上传 / 拖拽到此区域
          </span>
        </button>
      ) : (
        <div className="relative">
          <img
            ref={imageRef}
            src={imageUrl}
            alt={title}
            className="h-[360px] w-full object-contain bg-slate-950/70"
            onClick={() => inputRef.current?.click()}
          />
          <canvas
            ref={canvasRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
        </div>
      )}
    </section>
  );
}
