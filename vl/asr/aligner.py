"""转录-说话人对齐 - 将 ASR 片段与说话人标签匹配"""

from vl.core.logging import get_logger
from vl.core.models.transcript import TranscriptSegment, DiarizationSegment

logger = get_logger()


class Aligner:
    """转录片段与说话人对齐器"""

    def align(
        self,
        segments: list[TranscriptSegment],
        diarization: list[DiarizationSegment],
    ) -> list[TranscriptSegment]:
        """将转录片段分配到对应的说话人"""
        if not diarization:
            logger.warning("无说话人识别结果，跳过对齐")
            return segments

        aligned = 0
        for seg in segments:
            best_speaker = None
            best_overlap = 0.0

            for dia in diarization:
                # 计算时间重叠
                overlap = max(
                    0.0,
                    min(seg.end_time, dia.end_time) - max(seg.start_time, dia.start_time),
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = dia.speaker

            if best_speaker and best_overlap > 0:
                seg.speaker_id = best_speaker
                aligned += 1

        logger.info(f"说话人对齐完成: {aligned}/{len(segments)} 个片段已匹配")
        return segments
