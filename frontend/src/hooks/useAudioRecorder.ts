"use client";

import { useState, useRef, useCallback } from "react";

/**
 * 麦克风录音 hook (MediaRecorder API).
 *
 * 用法: startRecording() → stopRecording() → transcribe(blob) → 文字
 */
export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.start();
    mediaRecorderRef.current = recorder;
    setIsRecording(true);
  }, []);

  const stopRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder) return resolve(null);
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        recorder.stream.getTracks().forEach((t) => t.stop());
        setIsRecording(false);
        resolve(blob);
      };
      recorder.stop();
    });
  }, []);

  const transcribe = useCallback(async (blob: Blob): Promise<string> => {
    setIsTranscribing(true);
    const t0 = Date.now();
    try {
      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");
      const res = await fetch("/api/asr", { method: "POST", body: formData });
      const data = await res.json();
      console.log(`[ASR] ${Date.now() - t0}ms`);
      return data.text || "";
    } finally {
      setIsTranscribing(false);
    }
  }, []);

  return { isRecording, isTranscribing, startRecording, stopRecording, transcribe };
}
