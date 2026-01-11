"""
SIFT-based Stationary Object Detection for BeeMonitor

Enhanced with movement validation to prevent false DL detections from
corrupting templates. Drop-in replacement for beemonitor/detection/sift_detector.py

Usage:
    # Same interface as before
    sift = SIFTDetector(min_keypoints=3)
    
    # Enhanced initialization (with movement validation)
    num_templates = sift.initialize_from_video(
        video_path='bee_hotel.mp4',
        yolo_detector=yolo,
        num_frames=100,
        start_frame=100,
        min_confidence=0.7
    )
    
    # Detection works the same
    detections = sift.detect(frame, use_templates=True)
    
    # Backward compatible - can disable robustness features
    sift = SIFTDetector(require_movement=False)  # Legacy mode
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle

from beemonitor.detection.base_detector import BaseDetector, Detection


class SIFTDetector(BaseDetector):
    """
    SIFT-based detector for stationary objects (e.g., bees that aren't moving).
    
    Enhanced with movement validation to ensure only moving objects
    are learned as templates, preventing static objects (nest holes)
    from being tracked.
    """
    
    def __init__(
        self,
        min_keypoints: int = 3,
        match_threshold: float = 0.7,
        use_templates: bool = True,
        # Movement validation (ROBUST - NEW)
        require_movement: bool = True,
        movement_threshold: float = 20.0,
        template_quality_threshold: float = 50.0,
        spatial_grid_size: int = 100,
        max_templates_per_region: int = 2,
        # Runtime validation (ROBUST - NEW)
        enable_runtime_validation: bool = True,
        stationary_window: int = 30,
        stationary_threshold: float = 10.0
    ):
        """
        Initialize SIFT detector.
        
        Args:
            min_keypoints: Minimum keypoints for valid detection
            match_threshold: Ratio test threshold for feature matching  
            use_templates: Whether to use learned templates
            require_movement: Require movement during template learning (ROBUST)
            movement_threshold: Minimum movement (pixels) for valid template
            template_quality_threshold: Minimum quality score (0-100)
            spatial_grid_size: Grid size for spatial diversity check
            max_templates_per_region: Max templates per grid cell
            enable_runtime_validation: Filter stationary matches at runtime
            stationary_window: Frames to track for stationary check
            stationary_threshold: Max movement to be stationary
        """
        super().__init__()
        
        self.min_keypoints = min_keypoints
        self.match_threshold = match_threshold
        self.use_templates = use_templates
        
        # Robustness parameters
        self.require_movement = require_movement
        self.movement_threshold = movement_threshold
        self.template_quality_threshold = template_quality_threshold
        self.spatial_grid_size = spatial_grid_size
        self.max_templates_per_region = max_templates_per_region
        self.enable_runtime_validation = enable_runtime_validation
        self.stationary_window = stationary_window
        self.stationary_threshold = stationary_threshold
        
        # SIFT detector
        self.sift = cv2.SIFT_create()
        
        # Templates
        self.templates: List[Dict] = []
        
        # Runtime tracking
        self.match_history: List[Dict] = []
    
    def get_source_name(self) -> str:
        """Get detector source name."""
        return 'sift'
    
    # ========================================================================
    # Template Learning (ENHANCED - Same method name, robust implementation)
    # ========================================================================
    
    def initialize_from_video(
        self,
        video_path: str,
        yolo_detector,
        num_frames: int = 100,
        start_frame: int = 0,
        min_confidence: float = 0.7
    ) -> int:
        """
        Learn SIFT templates from video with movement validation.
        
        Enhanced to only learn from moving objects, preventing static
        objects (nest holes) from being learned as bee templates.
        
        Args:
            video_path: Path to video file
            yolo_detector: YOLODetector instance
            num_frames: Number of frames to process
            start_frame: Starting frame number
            min_confidence: Minimum YOLO confidence
        
        Returns:
            Number of templates learned
        """
        if not self.require_movement:
            # Legacy mode (no movement check)
            return self._initialize_legacy(
                video_path, yolo_detector, num_frames, 
                start_frame, min_confidence
            )
        
        print(f"\nSIFT initialization with movement validation:")
        print(f"  Video: {Path(video_path).name}")
        print(f"  Frames: {start_frame} to {start_frame + num_frames}")
        print(f"  Movement threshold: {self.movement_threshold}px")
        
        # Track detections across frames
        trajectories = self._track_detections(
            video_path, yolo_detector, num_frames, 
            start_frame, min_confidence
        )
        
        print(f"\nTracked {len(trajectories)} objects")
        
        # Filter by movement
        moving = [
            t for t in trajectories
            if self._calc_movement(t) >= self.movement_threshold
        ]
        
        stationary = len(trajectories) - len(moving)
        print(f"  Moving: {len(moving)}")
        print(f"  Stationary (filtered): {stationary}")
        
        if not moving:
            print("\n⚠️  WARNING: No moving objects found!")
            print("   Try different frame range or lower movement_threshold")
            return 0
        
        # Learn templates
        initial = len(self.templates)
        for traj in moving:
            self._learn_template(traj, video_path)
        
        # Filter quality and spatial duplicates
        self._filter_quality()
        self._filter_spatial()
        
        final = len(self.templates)
        print(f"\n✓ Learned {final} templates")
        
        if final == 0:
            print("   ⚠️  No valid templates - SIFT won't work")
        elif final > 50:
            print(f"   ⚠️  {final} templates is high - may include nest holes")
        
        return final
    
    def initialize_with_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        min_confidence: float = 0.7,
        frame_num: int = 0
    ) -> int:
        """Initialize from single frame detections."""
        added = 0
        
        for det in detections:
            if det.confidence < min_confidence:
                continue
            
            x1, y1, x2, y2 = det.bbox
            # Convert to integers (YOLO returns floats)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            region = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            kp, desc = self.sift.detectAndCompute(gray, None)
            
            if desc is None or len(kp) < self.min_keypoints:
                continue
            
            self.templates.append({
                'keypoints': kp,
                'descriptors': desc,
                'quality_score': 70.0,
                'spatial_region': self._spatial_region(det.centroid),
                'frame_num': frame_num,  # Store source frame
                'bbox': (x1, y1, x2, y2)  # Store bbox
            })
            added += 1
        
        return added
    
    # ========================================================================
    # Detection (ENHANCED - Same method name, robust implementation)
    # ========================================================================
    
    def detect(
        self,
        frame: np.ndarray,
        use_templates: bool = True,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Detection]:
        """
        Detect objects using SIFT with runtime validation.
        
        Args:
            frame: Input frame (BGR)
            use_templates: Use learned templates
            roi: Optional ROI (x1, y1, x2, y2)
        
        Returns:
            List of detections
        """
        if not use_templates or len(self.templates) == 0:
            return self._detect_clustering(frame, roi)
        
        # Extract SIFT from frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if roi:
            x1, y1, x2, y2 = roi
            gray_roi = gray[y1:y2, x1:x2]
            kp, desc = self.sift.detectAndCompute(gray_roi, None)
            # Adjust positions
            for k in kp:
                k.pt = (k.pt[0] + x1, k.pt[1] + y1)
        else:
            kp, desc = self.sift.detectAndCompute(gray, None)
        
        if desc is None:
            return []
        
        # Match templates
        detections = self._match_templates(kp, desc)
        
        # Runtime validation
        if self.enable_runtime_validation:
            detections = self._validate_runtime(detections)
        
        return detections
    
    # ========================================================================
    # Movement Validation (ROBUST CORE)
    # ========================================================================
    
    def _track_detections(
        self,
        video_path: str,
        yolo_detector,
        num_frames: int,
        start_frame: int,
        min_confidence: float
    ) -> List[Dict]:
        """Track YOLO detections to build trajectories."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        trajs = {}
        next_id = 0
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            dets = yolo_detector.detect(frame, conf=min_confidence)
            
            if i % 20 == 0:
                print(f"  Frame {start_frame + i}: {len(dets)} dets")
            
            if i == 0:
                # Create trajectories
                for d in dets:
                    trajs[next_id] = {
                        'positions': [d.centroid],
                        'bboxes': [d.bbox],
                        'frames': [start_frame + i]
                    }
                    next_id += 1
            else:
                # Match to existing
                matched = self._match_simple(dets, trajs, start_frame + i)
                
                # New trajectories for unmatched
                for d in dets:
                    if d not in matched:
                        trajs[next_id] = {
                            'positions': [d.centroid],
                            'bboxes': [d.bbox],
                            'frames': [start_frame + i]
                        }
                        next_id += 1
        
        cap.release()
        
        # Return valid trajectories
        return [t for t in trajs.values() if len(t['positions']) >= 3]
    
    def _match_simple(
        self,
        dets: List[Detection],
        trajs: Dict,
        frame_num: int
    ) -> List[Detection]:
        """Simple nearest-neighbor matching."""
        matched = []
        
        for d in dets:
            best_id = None
            best_dist = 100.0
            
            for tid, traj in trajs.items():
                last = traj['positions'][-1]
                dist = np.sqrt(
                    (d.centroid[0] - last[0])**2 +
                    (d.centroid[1] - last[1])**2
                )
                
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            
            if best_id is not None:
                trajs[best_id]['positions'].append(d.centroid)
                trajs[best_id]['bboxes'].append(d.bbox)
                trajs[best_id]['frames'].append(frame_num)
                matched.append(d)
        
        return matched
    
    def _calc_movement(self, traj: Dict) -> float:
        """Calculate total trajectory movement."""
        pos = traj['positions']
        if len(pos) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(pos)):
            dx = pos[i][0] - pos[i-1][0]
            dy = pos[i][1] - pos[i-1][1]
            total += np.sqrt(dx**2 + dy**2)
        
        return total
    
    def _learn_template(self, traj: Dict, video_path: str) -> bool:
        """Extract template from trajectory."""
        # Use middle frame
        mid = len(traj['frames']) // 2
        frame_num = traj['frames'][mid]
        bbox = traj['bboxes'][mid]
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
        
        # Extract region
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        region = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)
        
        if desc is None or len(kp) < self.min_keypoints:
            return False
        
        # Calculate quality
        quality = self._calc_quality(traj, kp)
        
        # Spatial region
        avg_pos = np.mean(traj['positions'], axis=0)
        region_id = self._spatial_region(tuple(avg_pos))
        
        self.templates.append({
            'keypoints': kp,
            'descriptors': desc,
            'trajectory': traj,
            'quality_score': quality,
            'spatial_region': region_id
        })
        
        return True
    
    def _calc_quality(self, traj: Dict, kp) -> float:
        """Score template quality (0-100)."""
        score = 0.0
        
        # Movement (30 pts)
        movement = self._calc_movement(traj)
        score += min(30, movement / 200 * 30)
        
        # Keypoints (25 pts)
        score += min(25, len(kp) / 10 * 25)
        
        # Trajectory length (20 pts)
        score += min(20, len(traj['positions']) / 20 * 20)
        
        # Variance (25 pts)
        pos = traj['positions']
        if len(pos) > 2:
            xs = [p[0] for p in pos]
            ys = [p[1] for p in pos]
            var = np.var(xs) + np.var(ys)
            if 10 < var < 2000:
                score += 25
        
        return score
    
    def _filter_quality(self):
        """Remove low quality templates."""
        self.templates = [
            t for t in self.templates
            if t['quality_score'] >= self.template_quality_threshold
        ]
    
    def _spatial_region(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Get spatial grid cell."""
        return (
            int(pos[0] // self.spatial_grid_size),
            int(pos[1] // self.spatial_grid_size)
        )
    
    def _filter_spatial(self):
        """Keep best templates per region."""
        regions = {}
        for t in self.templates:
            r = t['spatial_region']
            if r not in regions:
                regions[r] = []
            regions[r].append(t)
        
        filtered = []
        for r, temps in regions.items():
            sorted_temps = sorted(temps, key=lambda x: x['quality_score'], reverse=True)
            filtered.extend(sorted_temps[:self.max_templates_per_region])
        
        self.templates = filtered
    
    # ========================================================================
    # Runtime Validation (ROBUST)
    # ========================================================================
    
    def _validate_runtime(self, dets: List[Detection]) -> List[Detection]:
        """Filter stationary detections."""
        validated = []
        
        for d in dets:
            self.match_history.append({
                'pos': d.centroid,
                'frame': len(self.match_history)
            })
            
            if len(self.match_history) > self.stationary_window:
                self.match_history = self.match_history[-self.stationary_window:]
            
            # Check if stationary
            if len(self.match_history) >= self.stationary_window:
                similar = sum(
                    1 for h in self.match_history[-self.stationary_window:]
                    if np.sqrt(
                        (h['pos'][0] - d.centroid[0])**2 +
                        (h['pos'][1] - d.centroid[1])**2
                    ) < self.stationary_threshold
                )
                
                if similar > self.stationary_window * 0.8:
                    continue  # Skip stationary
            
            validated.append(d)
        
        return validated
    
    # ========================================================================
    # Legacy/Backward Compatibility
    # ========================================================================
    
    def _initialize_legacy(
        self,
        video_path: str,
        yolo_detector,
        num_frames: int,
        start_frame: int,
        min_confidence: float
    ) -> int:
        """Legacy initialization (no movement check)."""
        print("\nSIFT initialization (legacy mode)")
        print(f"  Video path: {video_path}")
        print(f"  Processing {num_frames} frames starting from frame {start_frame}")
        print(f"  YOLO detector: {yolo_detector}")
        print(f"  Min confidence (for template filtering): {min_confidence}")
        
        # Test if video opens
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ ERROR: Could not open video: {video_path}")
            return 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        total_dets = 0
        frames_processed = 0
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"  ⚠️  Could not read frame {start_frame + i} (video ended?)")
                break
            
            frames_processed += 1
            
            # Use YOLO's own confidence threshold (don't override!)
            dets = yolo_detector.detect(frame)  # No conf parameter!
            total_dets += len(dets)
            
            if i % 50 == 0 or len(dets) > 0:
                print(f"  Frame {start_frame + i}: {len(dets)} detections (total so far: {total_dets})")
            
            if len(dets) > 0:
                added = self.initialize_with_detections(
                    frame, dets, min_confidence, frame_num=start_frame + i
                )
                if i % 50 == 0:
                    print(f"    → Added {added} templates (total templates: {len(self.templates)})")
        
        cap.release()
        
        print(f"\n✓ Processed {frames_processed} frames, found {total_dets} total detections")
        print(f"✓ Learned {len(self.templates)} templates (legacy)")
        
        if total_dets == 0:
            print("\n❌ PROBLEM: YOLO found 0 detections!")
            print("   Possible causes:")
            print("   - YOLO detector not configured correctly")
            print("   - Confidence threshold too high")
            print("   - Wrong tracking classes")
            print("   - Video has no bees in this range")
        
        return len(self.templates)
    
    def _detect_clustering(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple] = None
    ) -> List[Detection]:
        """Clustering-based detection (original method)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)
        
        if desc is None or len(kp) < self.min_keypoints:
            return []
        
        # Spatial clustering
        from sklearn.cluster import DBSCAN
        
        pos = np.array([k.pt for k in kp])
        clustering = DBSCAN(eps=30, min_samples=self.min_keypoints).fit(pos)
        
        dets = []
        for cid in set(clustering.labels_):
            if cid == -1:
                continue
            
            cluster = pos[clustering.labels_ == cid]
            centroid = tuple(np.mean(cluster, axis=0))
            x1, y1 = int(cluster[:, 0].min() - 10), int(cluster[:, 1].min() - 10)
            x2, y2 = int(cluster[:, 0].max() + 10), int(cluster[:, 1].max() + 10)
            
            dets.append(Detection(
                bbox=(x1, y1, x2, y2),
                centroid=centroid,
                confidence=len(cluster) / 20.0,
                label='bee',
                source='sift',
                metadata={'num_keypoints': len(cluster)}
            ))
        
        return dets
    
    def _match_templates(self, kp, desc) -> List[Detection]:
        """Match against learned templates."""
        dets = []
        bf = cv2.BFMatcher()
        
        for template in self.templates:
            matches = bf.knnMatch(template['descriptors'], desc, k=2)
            
            # Ratio test
            good = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < self.match_threshold * n.distance:
                        good.append(m)
            
            if len(good) >= self.min_keypoints:
                # Get positions
                matched_pos = [kp[m.trainIdx].pt for m in good]
                centroid = tuple(np.mean(matched_pos, axis=0))
                xs = [p[0] for p in matched_pos]
                ys = [p[1] for p in matched_pos]
                x1, x2 = int(min(xs) - 20), int(max(xs) + 20)
                y1, y2 = int(min(ys) - 20), int(max(ys) + 20)
                
                dets.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    centroid=centroid,
                    confidence=len(good) / len(template['keypoints']),
                    label='bee',
                    source='sift',
                    metadata={
                        'num_matches': len(good),
                        'template_quality': template.get('quality_score', 0)
                    }
                ))
        
        return dets
    
    # ========================================================================
    # Template Management (Same interface as before)
    # ========================================================================
    
    def save_templates(self, filepath: str):
        """Save templates to file (convert KeyPoints to picklable format)."""
        # Convert KeyPoints to tuples (cv2.KeyPoint can't be pickled)
        serializable_templates = []
        for template in self.templates:
            kp = template['keypoints']
            # Convert each KeyPoint to tuple: (pt, size, angle, response, octave, class_id)
            kp_data = [(k.pt, k.size, k.angle, k.response, k.octave, k.class_id) for k in kp]
            
            serializable_templates.append({
                'keypoints': kp_data,  # List of tuples
                'descriptors': template['descriptors'],
                'quality_score': template.get('quality_score', 50.0),
                'spatial_region': template.get('spatial_region', None),
                'frame_num': template.get('frame_num', 0),  # Save frame number
                'bbox': template.get('bbox', None)  # Save bbox
            })
        
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_templates, f)
        print(f"✓ Saved {len(serializable_templates)} templates")
    
    def load_templates(self, filepath: str):
        """Load templates from file (reconstruct KeyPoints)."""
        with open(filepath, 'rb') as f:
            serializable_templates = pickle.load(f)
        
        # Reconstruct KeyPoints from tuples
        self.templates = []
        for template in serializable_templates:
            kp_data = template['keypoints']
            # Reconstruct each KeyPoint from tuple
            kp = [cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle, 
                               response=response, octave=octave, class_id=class_id)
                  for pt, size, angle, response, octave, class_id in kp_data]
            
            self.templates.append({
                'keypoints': kp,  # List of cv2.KeyPoint objects
                'descriptors': template['descriptors'],
                'quality_score': template.get('quality_score', 50.0),
                'spatial_region': template.get('spatial_region', None),
                'frame_num': template.get('frame_num', 0),  # Restore frame number
                'bbox': template.get('bbox', None)  # Restore bbox
            })
        
        print(f"✓ Loaded {len(self.templates)} templates")
    
    def get_num_templates(self) -> int:
        """Get number of learned templates."""
        if not self.templates:
            return 0
        return len(self.templates)
    
    def visualize_templates(
        self,
        video_path: str,
        output_dir: str,
        max_templates: int = 10
    ) -> int:
        """Visualize learned SIFT templates and save to directory.
        
        Args:
            video_path: Path to video file (to extract template frames)
            output_dir: Directory to save visualizations
            max_templates: Maximum number of templates to visualize
            
        Returns:
            Number of templates visualized
        """
        from pathlib import Path
        import os
        
        if not self.templates:
            print("⚠️  No templates to visualize")
            return 0
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sort templates by quality score (best first)
        sorted_templates = sorted(
            self.templates,
            key=lambda t: t.get('quality_score', 0),
            reverse=True
        )
        
        num_to_viz = min(len(sorted_templates), max_templates)
        print(f"Visualizing {num_to_viz} SIFT templates...")
        
        cap = cv2.VideoCapture(video_path)
        
        for i, template in enumerate(sorted_templates[:num_to_viz]):
            # Get frame and bbox from template (legacy mode)
            frame_num = template.get('frame_num', 0)
            bbox = template.get('bbox', None)
            
            if bbox is None:
                # Fallback for old trajectory format
                if 'trajectory' in template:
                    traj = template['trajectory']
                    mid_idx = len(traj['frames']) // 2
                    frame_num = traj['frames'][mid_idx]
                    bbox = traj['bboxes'][mid_idx]
                else:
                    continue  # Skip if no frame/bbox info
            
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Extract bbox region
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract crop
            crop = frame[y1:y2, x1:x2].copy()
            
            # Draw keypoints on crop
            keypoints = template['keypoints']
            crop_with_kp = cv2.drawKeypoints(
                crop,
                keypoints,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=(0, 255, 0)
            )
            
            # Create visualization with info
            quality = template.get('quality_score', 0)
            region = template.get('spatial_region', 'N/A')
            num_kp = len(keypoints)
            
            # Add text overlay
            h, w = crop_with_kp.shape[:2]
            info_text = [
                f"Template {i+1}/{num_to_viz}",
                f"Frame: {frame_num}",
                f"Quality: {quality:.2f}",
                f"Keypoints: {num_kp}",
                f"Region: {region}"
            ]
            
            # Create padded image for text
            padding = 120
            viz = np.zeros((h + padding, w, 3), dtype=np.uint8)
            viz[:h, :w] = crop_with_kp
            
            # Draw text
            y_offset = h + 20
            for text in info_text:
                cv2.putText(
                    viz,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
                y_offset += 20
            
            # Save visualization
            output_file = output_path / f"sift_template_{i+1:02d}_frame{frame_num}_q{quality:.2f}.jpg"
            cv2.imwrite(str(output_file), viz)
        
        cap.release()
        
        print(f"✓ Saved {num_to_viz} template visualizations to {output_dir}")
        return num_to_viz
    
    def get_num_templates(self) -> int:
        """Get number of templates."""
        return len(self.templates)
    
    def clear_templates(self):
        """Clear all templates."""
        self.templates = []
        self.match_history = []
    
    def get_template_statistics(self) -> Dict:
        """Get template statistics."""
        if not self.templates:
            return {'num_templates': 0}
        
        qualities = [t['quality_score'] for t in self.templates]
        kp_counts = [len(t['keypoints']) for t in self.templates]
        
        stats = {
            'num_templates': len(self.templates),
            'avg_quality': np.mean(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'avg_keypoints': np.mean(kp_counts)
        }
        
        # Add movement stats if available
        movements = []
        for t in self.templates:
            if 'trajectory' in t:
                movements.append(self._calc_movement(t['trajectory']))
        
        if movements:
            stats['avg_movement'] = np.mean(movements)
            stats['min_movement'] = np.min(movements)
            stats['max_movement'] = np.max(movements)
        
        return stats
    
    # ========================================================================
    # Configuration (Same interface)
    # ========================================================================
    
    def configure(self, **kwargs):
        """Configure detector parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def reset(self):
        """Reset detector state."""
        self.match_history = []