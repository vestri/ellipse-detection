#import exceptions

from ellipse_detection.segment_detector import SegmentDetector
from ellipse_detection.ellipse_candidate_maker import EllipseCandidateMaker
from ellipse_detection.ellipse_estimator import EllipseEstimator
from ellipse_detection.ellipse_merger import EllipseMerger

import cv2

class EllipseDetector(object):
    def __init__(self):
        pass

    def detect(self, image, debug_image=None):
        """Detect ellipse from image.

        Args:
            image: A numpy as array indicats gray scale image.

        Returns:
            Array of Ellipse instance that was detected from image.
        """

        if len(image.shape) != 2:
            raise exceptions.RuntimeException()

        seg_detector = SegmentDetector()
        segments = seg_detector.detect(image)

        if(debug_image is not None):
            image_segments = debug_image.copy()
            for segmentDir in segments:
                for segment in segmentDir:
                    segment.draw(image_segments)
        
            cv2.imshow('segments', image_segments)
            cv2.waitKey(0)


        ellipse_cand_maker = EllipseCandidateMaker()
        ellipse_cands = ellipse_cand_maker.make(segments, debug_image)

        ellipse_estimator = EllipseEstimator()
        ellipses = ellipse_estimator.estimate(ellipse_cands)

        ellipse_merger = EllipseMerger(image.shape[1], image.shape[0])
        ellipses = ellipse_merger.merge(ellipses)

        return ellipses
