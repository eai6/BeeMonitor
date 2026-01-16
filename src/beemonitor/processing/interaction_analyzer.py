"""Interaction Analyzer - Track proximity and interaction metrics.

Computes interaction metrics between:
- Tracks and other tracks (bee-to-bee interactions)
- Tracks and reference objects (bee-to-nest interactions)

Output metrics include:
- Time spent in proximity
- Number of interaction events
- Interaction duration statistics
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class InteractionEvent:
    """Single interaction event between two entities."""
    entity1_id: Any  # track_id or 'nest_X'
    entity2_id: Any
    start_frame: int
    end_frame: int
    min_distance: float
    avg_distance: float
    
    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class InteractionSummary:
    """Summary of all interactions for an entity."""
    entity_id: Any
    total_interaction_frames: int = 0
    num_interactions: int = 0
    unique_partners: int = 0
    interaction_partners: List[Any] = field(default_factory=list)
    avg_interaction_duration: float = 0.0
    max_interaction_duration: int = 0


class InteractionAnalyzer:
    """Analyzes interactions between tracks and reference objects.
    
    Computes proximity-based interaction metrics for behavioral analysis.
    """
    
    def __init__(
        self,
        proximity_threshold: float = 50.0,
        min_interaction_frames: int = 3,
        fps: float = 30.0
    ):
        """Initialize interaction analyzer.
        
        Args:
            proximity_threshold: Distance (pixels) to consider as "interacting"
            min_interaction_frames: Minimum frames to count as interaction
            fps: Video frame rate for time conversions
        """
        self.proximity_threshold = proximity_threshold
        self.min_interaction_frames = min_interaction_frames
        self.fps = fps
        
        logger.info(f"InteractionAnalyzer initialized: "
                   f"proximity={proximity_threshold}px, min_frames={min_interaction_frames}")
    
    def analyze_track_interactions(
        self,
        tracking_df: pd.DataFrame
    ) -> Tuple[List[InteractionEvent], pd.DataFrame]:
        """Analyze interactions between tracks (bee-to-bee).
        
        Args:
            tracking_df: DataFrame with columns [frame, track_id, cx, cy, ...]
            
        Returns:
            Tuple of (interaction_events, summary_df)
        """
        if tracking_df.empty:
            return [], pd.DataFrame()
        
        interactions = []
        
        # Group by frame
        frames = tracking_df.groupby('frame')
        
        # Track active interactions: (track1, track2) -> start_frame, distances
        active_interactions = {}
        
        for frame_num, frame_data in frames:
            track_ids = frame_data['track_id'].unique()
            
            # Get positions for all tracks in this frame
            positions = {}
            for _, row in frame_data.iterrows():
                positions[row['track_id']] = (row['cx'], row['cy'])
            
            # Check all pairs
            current_pairs = set()
            for i, t1 in enumerate(track_ids):
                for t2 in track_ids[i+1:]:
                    if t1 not in positions or t2 not in positions:
                        continue
                    
                    p1 = positions[t1]
                    p2 = positions[t2]
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    
                    pair = (min(t1, t2), max(t1, t2))
                    
                    if dist <= self.proximity_threshold:
                        current_pairs.add(pair)
                        
                        if pair not in active_interactions:
                            # Start new interaction
                            active_interactions[pair] = {
                                'start_frame': frame_num,
                                'distances': [dist]
                            }
                        else:
                            # Continue interaction
                            active_interactions[pair]['distances'].append(dist)
            
            # Check for ended interactions
            ended_pairs = []
            for pair in active_interactions:
                if pair not in current_pairs:
                    ended_pairs.append(pair)
            
            # Finalize ended interactions
            for pair in ended_pairs:
                info = active_interactions.pop(pair)
                duration = len(info['distances'])
                
                if duration >= self.min_interaction_frames:
                    interactions.append(InteractionEvent(
                        entity1_id=pair[0],
                        entity2_id=pair[1],
                        start_frame=info['start_frame'],
                        end_frame=info['start_frame'] + duration - 1,
                        min_distance=min(info['distances']),
                        avg_distance=np.mean(info['distances'])
                    ))
        
        # Finalize any remaining active interactions
        for pair, info in active_interactions.items():
            duration = len(info['distances'])
            if duration >= self.min_interaction_frames:
                interactions.append(InteractionEvent(
                    entity1_id=pair[0],
                    entity2_id=pair[1],
                    start_frame=info['start_frame'],
                    end_frame=info['start_frame'] + duration - 1,
                    min_distance=min(info['distances']),
                    avg_distance=np.mean(info['distances'])
                ))
        
        # Create summary DataFrame
        summary_df = self._create_interaction_summary(interactions, tracking_df)
        
        logger.info(f"Track interactions: {len(interactions)} events detected")
        
        return interactions, summary_df
    
    def analyze_reference_interactions(
        self,
        tracking_df: pd.DataFrame,
        reference_objects: Dict[str, Tuple[float, float, float, float]]
    ) -> Tuple[List[InteractionEvent], pd.DataFrame]:
        """Analyze interactions between tracks and reference objects (bee-to-nest).
        
        Args:
            tracking_df: DataFrame with columns [frame, track_id, cx, cy, ...]
            reference_objects: Dict mapping object_id to bbox (x1, y1, x2, y2)
            
        Returns:
            Tuple of (interaction_events, summary_df)
        """
        if tracking_df.empty or not reference_objects:
            return [], pd.DataFrame()
        
        interactions = []
        
        # Compute reference object centers
        ref_centers = {}
        for ref_id, bbox in reference_objects.items():
            x1, y1, x2, y2 = bbox
            ref_centers[ref_id] = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Group by frame
        frames = tracking_df.groupby('frame')
        
        # Track active interactions: (track_id, ref_id) -> start_frame, distances
        active_interactions = {}
        
        for frame_num, frame_data in frames:
            current_pairs = set()
            
            for _, row in frame_data.iterrows():
                track_id = row['track_id']
                track_pos = (row['cx'], row['cy'])
                
                for ref_id, ref_pos in ref_centers.items():
                    dist = np.sqrt(
                        (track_pos[0] - ref_pos[0])**2 + 
                        (track_pos[1] - ref_pos[1])**2
                    )
                    
                    pair = (track_id, ref_id)
                    
                    if dist <= self.proximity_threshold:
                        current_pairs.add(pair)
                        
                        if pair not in active_interactions:
                            active_interactions[pair] = {
                                'start_frame': frame_num,
                                'distances': [dist]
                            }
                        else:
                            active_interactions[pair]['distances'].append(dist)
            
            # Check for ended interactions
            ended_pairs = []
            for pair in active_interactions:
                if pair not in current_pairs:
                    ended_pairs.append(pair)
            
            for pair in ended_pairs:
                info = active_interactions.pop(pair)
                duration = len(info['distances'])
                
                if duration >= self.min_interaction_frames:
                    interactions.append(InteractionEvent(
                        entity1_id=pair[0],  # track_id
                        entity2_id=pair[1],  # ref_id
                        start_frame=info['start_frame'],
                        end_frame=info['start_frame'] + duration - 1,
                        min_distance=min(info['distances']),
                        avg_distance=np.mean(info['distances'])
                    ))
        
        # Finalize remaining
        for pair, info in active_interactions.items():
            duration = len(info['distances'])
            if duration >= self.min_interaction_frames:
                interactions.append(InteractionEvent(
                    entity1_id=pair[0],
                    entity2_id=pair[1],
                    start_frame=info['start_frame'],
                    end_frame=info['start_frame'] + duration - 1,
                    min_distance=min(info['distances']),
                    avg_distance=np.mean(info['distances'])
                ))
        
        # Create summary
        summary_df = self._create_reference_summary(interactions, reference_objects)
        
        logger.info(f"Reference interactions: {len(interactions)} events detected")
        
        return interactions, summary_df
    
    def _create_interaction_summary(
        self,
        interactions: List[InteractionEvent],
        tracking_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create summary DataFrame for track-to-track interactions."""
        if not interactions:
            return pd.DataFrame(columns=[
                'track_id', 'total_interaction_frames', 'num_interactions',
                'unique_partners', 'avg_duration_frames', 'avg_duration_seconds',
                'max_duration_frames', 'partners'
            ])
        
        # Aggregate by track
        track_stats = defaultdict(lambda: {
            'frames': 0, 'count': 0, 'partners': set(), 'durations': []
        })
        
        for event in interactions:
            for tid in [event.entity1_id, event.entity2_id]:
                partner = event.entity2_id if tid == event.entity1_id else event.entity1_id
                track_stats[tid]['frames'] += event.duration_frames
                track_stats[tid]['count'] += 1
                track_stats[tid]['partners'].add(partner)
                track_stats[tid]['durations'].append(event.duration_frames)
        
        rows = []
        for track_id, stats in track_stats.items():
            rows.append({
                'track_id': track_id,
                'total_interaction_frames': stats['frames'],
                'num_interactions': stats['count'],
                'unique_partners': len(stats['partners']),
                'avg_duration_frames': np.mean(stats['durations']) if stats['durations'] else 0,
                'avg_duration_seconds': np.mean(stats['durations']) / self.fps if stats['durations'] else 0,
                'max_duration_frames': max(stats['durations']) if stats['durations'] else 0,
                'partners': list(stats['partners'])
            })
        
        return pd.DataFrame(rows)
    
    def _create_reference_summary(
        self,
        interactions: List[InteractionEvent],
        reference_objects: Dict
    ) -> pd.DataFrame:
        """Create summary DataFrame for track-to-reference interactions."""
        if not interactions:
            return pd.DataFrame(columns=[
                'track_id', 'reference_id', 'total_frames', 'num_visits',
                'avg_visit_frames', 'avg_visit_seconds', 'max_visit_frames',
                'min_distance', 'avg_distance'
            ])
        
        # Aggregate by (track, reference) pair
        pair_stats = defaultdict(lambda: {
            'frames': 0, 'count': 0, 'durations': [], 'distances': []
        })
        
        for event in interactions:
            key = (event.entity1_id, event.entity2_id)
            pair_stats[key]['frames'] += event.duration_frames
            pair_stats[key]['count'] += 1
            pair_stats[key]['durations'].append(event.duration_frames)
            pair_stats[key]['distances'].append(event.avg_distance)
        
        rows = []
        for (track_id, ref_id), stats in pair_stats.items():
            rows.append({
                'track_id': track_id,
                'reference_id': ref_id,
                'total_frames': stats['frames'],
                'num_visits': stats['count'],
                'avg_visit_frames': np.mean(stats['durations']),
                'avg_visit_seconds': np.mean(stats['durations']) / self.fps,
                'max_visit_frames': max(stats['durations']),
                'min_distance': min(stats['distances']),
                'avg_distance': np.mean(stats['distances'])
            })
        
        return pd.DataFrame(rows)
    
    def get_interaction_timeline(
        self,
        interactions: List[InteractionEvent],
        total_frames: int
    ) -> pd.DataFrame:
        """Create frame-by-frame interaction timeline.
        
        Args:
            interactions: List of interaction events
            total_frames: Total frames in video
            
        Returns:
            DataFrame with columns [frame, num_interactions, interacting_pairs]
        """
        timeline = []
        
        for frame in range(total_frames):
            active = []
            for event in interactions:
                if event.start_frame <= frame <= event.end_frame:
                    active.append((event.entity1_id, event.entity2_id))
            
            timeline.append({
                'frame': frame,
                'num_interactions': len(active),
                'interacting_pairs': active
            })
        
        return pd.DataFrame(timeline)
    
    def to_csv(
        self,
        interactions: List[InteractionEvent],
        output_path: str,
        interaction_type: str = 'track'
    ):
        """Save interactions to CSV.
        
        Args:
            interactions: List of interaction events
            output_path: Path to save CSV
            interaction_type: 'track' or 'reference'
        """
        rows = []
        for event in interactions:
            rows.append({
                'entity1_id': event.entity1_id,
                'entity2_id': event.entity2_id,
                'start_frame': event.start_frame,
                'end_frame': event.end_frame,
                'duration_frames': event.duration_frames,
                'duration_seconds': event.duration_frames / self.fps,
                'min_distance': event.min_distance,
                'avg_distance': event.avg_distance,
                'interaction_type': interaction_type
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(interactions)} interactions to {output_path}")


def nests_to_reference_objects(nests: List[Dict]) -> Dict[str, Tuple[float, float, float, float]]:
    """Convert nest detections to reference objects format.
    
    Args:
        nests: List of nest dicts with 'bbox' or coordinate keys
        
    Returns:
        Dict mapping nest_id to bbox
    """
    reference_objects = {}
    
    for i, nest in enumerate(nests):
        nest_id = nest.get('id', nest.get('nest_id', f'nest_{i+1}'))
        
        if 'bbox' in nest:
            bbox = nest['bbox']
        elif all(k in nest for k in ['x1', 'y1', 'x2', 'y2']):
            bbox = (nest['x1'], nest['y1'], nest['x2'], nest['y2'])
        elif all(k in nest for k in ['x', 'y', 'width', 'height']):
            x, y, w, h = nest['x'], nest['y'], nest['width'], nest['height']
            bbox = (x, y, x + w, y + h)
        else:
            logger.warning(f"Cannot parse nest {i}: {nest}")
            continue
        
        reference_objects[str(nest_id)] = bbox
    
    return reference_objects