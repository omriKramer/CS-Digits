import operator

import numpy as np
from scipy import ndimage
from skimage import morphology, measure, graph

SIZE = 28


def max_score_point(score, mask):
    score_map = np.fromfunction(score, mask.shape)
    score_map = (score_map - score_map.min()) * mask
    index = np.argmax(score_map)
    index = np.unravel_index(index, mask.shape)
    return index


def segment_around_point(point, mask, length=5):
    i, j = int(round(point[0])), int(round(point[1]))
    segmentation = np.zeros_like(mask, dtype=bool)
    dx = length // 2
    x0 = max(0, i - dx)
    x1 = min(SIZE, i + dx + 1)
    y0 = max(0, j - dx)
    y1 = min(SIZE, j + dx + 1)
    segmentation[x0:x1, y0:y1] = True
    segmentation = segmentation & mask.astype(bool)
    return segmentation


def segment_path(path, digit_seg, width=3):
    dx = width // 2
    mask = np.zeros_like(digit_seg)
    for i, j in path:
        mask[i - dx: i + dx + 1, j - dx:j + dx + 1] = True
    mask *= digit_seg
    return mask


def find_center(blanks):
    labels = morphology.label(blanks, connectivity=2)
    props = measure.regionprops(labels, coordinates='xy')
    if len(props) > 1:
        props.sort(key=operator.attrgetter('area'), reverse=True)
        return props[1].centroid

    raise ValueError('Failed to find center')


def validate_features(features):
    to_delete = [key for key, segmentation in features.items() if not segmentation.any()]
    for key in to_delete:
        del features[key]


class FourFeatures:

    def __init__(self, image):
        digit_seg = image > 120
        self._skeleton = morphology.skeletonize(digit_seg)
        morphology.remove_small_objects(self._skeleton, min_size=8, connectivity=2, in_place=True)
        self._find_points()
        self.features = {
            'top_right': segment_around_point(self.top_right_pt, digit_seg),
            'top_left': segment_around_point(self.top_left_pt, digit_seg),
            'middle_left': segment_around_point(self.middle_left_pt, digit_seg),
            'middle_right': segment_around_point(self.middle_right_pt, digit_seg),
            'bottom': segment_around_point(self.bottom_pt, digit_seg)
        }
        validate_features(self.features)

    def _find_points(self):
        points = np.argwhere(self._skeleton)
        self.bottom_pt = max(points, key=operator.itemgetter(0, 1))

        self.top_right_pt = max_score_point(lambda i, j: -1 * ((SIZE - j) ** 2 + i ** 2), self._skeleton)

        top_left = max_score_point(lambda i, j: SIZE - j + SIZE - i, self._skeleton)
        self.top_left_pt = self.climb_up(top_left)

        middle_left = max_score_point(lambda i, j: SIZE / 2 - 0.5 * abs(SIZE / 2 - i) + SIZE - j, self._skeleton)
        self.middle_left_pt = self.climb_down(middle_left)

        has_left_neighbor = np.roll(self._skeleton, 1, axis=1)
        has_left_neighbor += np.roll(self._skeleton, 2, axis=1)
        has_left_neighbor += np.roll(self._skeleton, 3, axis=1)
        self.middle_right_pt = max_score_point(lambda i, j: j - abs(i - middle_left[0]),
                                               has_left_neighbor * self._skeleton)

    def climb_up(self, start):
        i, j = start
        while True:
            if self._skeleton[i - 1, j + 1]:
                i -= 1
                j += 1
            elif self._skeleton[i - 1, j]:
                i -= 1
            elif self._skeleton[i, j + 1]:
                j += 1
            else:
                return i, j

    def climb_down(self, start):
        i, j = start
        while True:
            if self._skeleton[i + 1, j]:
                i += 1
            elif self._skeleton[i + 1, j + 1]:
                i += 1
                j += 1
            elif self._skeleton[i + 1, j - 1]:
                i += 1
                j -= 1
            else:
                return i, j


class EightFeatures:

    def __init__(self, image):
        for thresh in [120, 50, 20, 200]:
            self.blanks = np.where(image > thresh, 0, 1)
            labels = morphology.label(self.blanks, connectivity=1)
            props = measure.regionprops(labels, coordinates='xy')
            if len(props) > 2:
                props.sort(key=operator.attrgetter('area'), reverse=True)
                self.bottom_pt, self.top_pt = props[1].centroid, props[2].centroid
                if self.bottom_pt[0] < self.top_pt[0]:
                    self.bottom_pt, self.top_pt = self.top_pt, self.bottom_pt
                break
        else:
            raise ValueError('Failed to find centers in image')

        self.features = {
            'top': segment_around_point(self.top_pt, self.blanks, 3),
            'bottom': segment_around_point(self.bottom_pt, self.blanks, 3),
        }
        validate_features(self.features)


class FiveFeatures:

    def __init__(self, image):
        segmentation = image > 60
        self._skeleton = morphology.skeletonize(segmentation)

        self.top_right = max_score_point(lambda i, j: j + (SIZE - i), self._skeleton)
        self.top_left = self.climb_left(self.top_right)
        mask = np.zeros_like(image, dtype=bool)
        mask[:self.top_left[0] + 1] = True
        top = segmentation & mask

        self.bottom_left = max_score_point(lambda i, j: i + (SIZE - j), self._skeleton)
        self.end_circle = self.circle(self.bottom_left)
        mask = np.zeros_like(image, dtype=bool)
        mask[self.end_circle[0]:] = True
        bottom = segmentation & mask

        self.features = {
            'top': top,
            'bottom': bottom,
        }
        validate_features(self.features)

    def climb_left(self, start):
        i, j = start
        while True:
            if self._skeleton[i, j - 1]:
                j -= 1
            elif self._skeleton[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif self._skeleton[i + 1, j - 1]:
                i += 1
                j -= 1
            else:
                return i, j

    def circle(self, start):
        i, j = start
        previous_up = False
        while True:
            if self._skeleton[i, j + 1]:
                j += 1
                previous_up = False
            elif i < SIZE - 1 and self._skeleton[i + 1, j + 1]:
                i += 1
                j += 1
                previous_up = False
            elif self._skeleton[i - 1, j + 1]:
                i -= 1
                j += 1
                previous_up = False
            elif self._skeleton[i - 1, j] and not previous_up:
                previous_up = True
                i -= 1
            else:
                break

        if self._skeleton[i - 1, j - 1]:
            i -= 1
            j -= 1

        while True:
            if self._skeleton[i - 2, j]:
                i -= 2
            elif self._skeleton[i - 1, j]:
                i -= 2
            else:
                break

        previous_up = False
        while True:
            if self._skeleton[i - 1, j - 1]:
                i -= 1
                j -= 1
                previous_up = False
            elif self._skeleton[i, j - 1]:
                j -= 1
                previous_up = False
            elif self._skeleton[i - 1, j] and not previous_up:
                previous_up = True
                i -= 1
            elif previous_up:
                return i + 1, j
            else:
                return i, j


class OneFeatures:

    def __init__(self, image):
        digit_seg = image > 60
        self._skeleton = morphology.skeletonize(digit_seg)
        morphology.remove_small_objects(self._skeleton, min_size=8, connectivity=2, in_place=True)
        top_pt = max_score_point(lambda i, j: SIZE - i, self._skeleton)
        self.top_pt = self.find_center(top_pt)
        self.bottom_pt = self.climb_down()
        self.features = {
            'top': segment_around_point(self.top_pt, digit_seg),
            'bottom': segment_around_point(self.bottom_pt, digit_seg)
        }

    def climb_down(self):
        i, j = self.top_pt
        while True:
            if self._skeleton[i + 1, j]:
                i += 1
            elif self._skeleton[i + 1, j + 1]:
                i += 1
                j += 1
            elif self._skeleton[i + 1, j - 1]:
                i += 1
                j -= 1
            else:
                return i, j

    def find_center(self, start):
        i, j = start
        mask = np.roll(self._skeleton, -1, axis=0)
        mask *= self._skeleton
        index = np.argmax(mask[i])
        if index != 0:
            return i, index
        return i, j


class ZeroFeatures:

    def __init__(self, image):
        thresh = 50
        try:
            self.blanks = np.where(image > thresh, 0, 1)
            self.center_pt = find_center(self.blanks)
        except ValueError:
            self.blanks = np.where(morphology.closing(image) > thresh, 0, 1)
            self.center_pt = find_center(self.blanks)

        self.features = {
            'center': segment_around_point(self.center_pt, self.blanks, 3)
        }


class TwoFeatures:

    def __init__(self, image):
        digit_seg = image > 20
        self._skeleton = morphology.skeletonize(digit_seg)
        top_left = max_score_point(lambda x, y: -1 * (x ** 2 + y ** 2), self._skeleton)
        self._find_paths(top_left)
        self.features = {
            'top': segment_path(self.top_path, digit_seg),
            'center': segment_path(self.center_path, digit_seg),
            'bottom': segment_path(self.bottom_path, digit_seg),
        }

    def _find_paths(self, start):
        i, j = start
        top1 = []
        while True:
            top1.append((i, j))
            if self._skeleton[i + 1, j] and j > 14:
                break

            if self._skeleton[i - 1, j + 1]:
                i -= 1
                j += 1
            elif self._skeleton[i, j + 1]:
                j += 1
            elif self._skeleton[i + 1, j + 1]:
                i += 1
                j += 1
            else:
                break

        top_shared = []
        while True:
            if self._skeleton[i, j + 1]:
                j += 1
                top_shared.append((i, j + 1))
            elif self._skeleton[i - 1, j + 1]:
                i -= 1
                j += 1
                top_shared.append((i, j))
            else:
                break

        self.top_path = top1 + top_shared
        if top_shared:
            top_shared.reverse()
            self.center_path = top_shared
        else:
            self.center_path = [top1[-1]]

        i, j = self.center_path[-1]
        bottom_shared = []
        while True:
            if (self._skeleton[i - 1, j] and j < 9) or bottom_shared:
                bottom_shared.append((i, j))

            if self._skeleton[i + 1, j - 1]:
                i += 1
                j -= 1
                self.center_path.append((i, j))
            elif self._skeleton[i + 1, j]:
                i += 1
                self.center_path.append((i, j))
            elif self._skeleton[i, j - 1] and (i, j - 1) not in self.top_path:
                j -= 1
                self.center_path.append((i, j))
            elif self._skeleton[i + 1, j + 1]:
                i += 1
                j += 1
                self.center_path.append((i, j))
            else:
                break

        if bottom_shared:
            bottom_shared.reverse()
            self.bottom_path = bottom_shared
        else:
            self.bottom_path = [self.center_path[-1]]

        i, j = self.bottom_path[-1]
        while True:
            if self._skeleton[i - 1, j - 1]:
                i -= 1
                j -= 1
                self.bottom_path.append((i, j))
            elif self._skeleton[i - 1, j] and (i - 1, j) not in self.center_path:
                i -= 1
                self.bottom_path.append((i, j))
            elif self._skeleton[i - 1, j + 1] and (i - 1, j + 1) not in self.center_path:
                i -= 1
                j += 1
                self.bottom_path.append((i, j))
            elif self._skeleton[i, j - 1]:
                j -= 1
                self.bottom_path.append((i, j))
            else:
                break

        end = max_score_point(lambda x, y: -1 * ((20 - x) ** 2 + (SIZE - y) ** 2), self._skeleton)
        g = np.where(self._skeleton, 1, 300)
        route, _ = graph.route_through_array(g, (i, j), end, geometric=False)
        self.bottom_path.extend(route[1:])

        tip = []
        i, j = start
        while True:
            if self._skeleton[i + 1, j] and (i + 1, j) not in self.bottom_path:
                i += 1
                tip.append((i, j))
            elif self._skeleton[i + 1, j - 1] and (i + 1, j - 1) not in self.bottom_path:
                i += 1
                j -= 1
                tip.append((i, j))
            elif self._skeleton[i, j - 1] and (i, j - 1) not in self.bottom_path:
                j -= 1
                tip.append((i, j))
            else:
                break

            tip.reverse()
            self.top_path = tip + self.top_path


class ThreeFeatures:

    def __init__(self, image):
        digit_seg = image > 20
        self._skeleton = morphology.skeletonize(digit_seg)
        morphology.remove_small_objects(self._skeleton, 8, connectivity=2, in_place=True)
        self.top_left = self._find_top_left()
        self.top_right = max_score_point(lambda x, y: -1 * (x ** 2 + (SIZE - y) ** 2), self._skeleton)
        self.bottom_left = self._find_bottom_left()
        self.bottom_right = max_score_point(lambda x, y: -1 * ((SIZE - x) ** 2 + (SIZE - y) ** 2), self._skeleton)
        self.center = self._find_center()

        g = np.where(self._skeleton, 1, 300)
        top1, _ = graph.route_through_array(g, self.top_left, self.top_right, geometric=False)
        top2, _ = graph.route_through_array(g, self.top_right, self.center, geometric=False)
        bottom1, _ = graph.route_through_array(g, self.bottom_left, self.bottom_right, geometric=False)
        bottom2, _ = graph.route_through_array(g, self.bottom_right, self.center, geometric=False)
        self.features = {
            'top_half': segment_path(top1 + top2, digit_seg),
            'bottom_half': segment_path(bottom1 + bottom2, digit_seg),
        }

    def _find_top_left(self):
        i, j = max_score_point(lambda x, y: -1 * (x ** 2 + y ** 2), self._skeleton)
        while True:
            if self._skeleton[i, j - 1]:
                i -= 1
            elif self._skeleton[i + 1, j - 1]:
                i += 1
                j -= 1
            elif self._skeleton[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif self._skeleton[i + 1, j]:
                i += 1
            else:
                return i, j

    def _find_bottom_left(self):
        i, j = max_score_point(lambda x, y: -1 * ((SIZE - x) ** 2 + y ** 2), self._skeleton)
        while True:
            if self._skeleton[i, j - 1]:
                i -= 1
            elif self._skeleton[i + 1, j - 1]:
                i += 1
                j -= 1
            elif self._skeleton[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif self._skeleton[i - 1, j]:
                i -= 1
            else:
                return i, j

    def _find_center(self):
        middle = (self.top_left[0] + self.bottom_left[0]) / 2, max(self.top_left[1], self.bottom_left[1]) + 3
        i, j = max_score_point(lambda x, y: -1 * ((middle[0] - x) ** 2 + (middle[1] - y) ** 2), self._skeleton)
        while True:
            if self._skeleton[i, j - 1]:
                j -= 1
            elif self._skeleton[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif self._skeleton[i + 1, j - 1]:
                i += 1
                j -= 1
            else:
                return i, j


class SixFeatures:

    def __init__(self, image):
        closed = morphology.closing(image)
        for thresh in range(240, 0, -30):
            try:
                self.blanks = np.where(image > thresh, 0, 1)
                self.center_pt = find_center(self.blanks)
            except ValueError:
                try:
                    self.blanks = np.where(closed > thresh, 0, 1)
                    self.center_pt = find_center(self.blanks)
                except ValueError:
                    continue

            break
        else:
            raise ValueError

        self._skeleton = morphology.skeletonize(image > 20)
        morphology.remove_small_objects(self._skeleton, min_size=4, connectivity=2, in_place=True)
        i, j = max_score_point(lambda x, y: SIZE - x, self._skeleton)
        while True:
            if j + 1 == SIZE:
                break

            if self._skeleton[i, j + 1]:
                j += 1
            elif i + 1 < SIZE and self._skeleton[i + 1, j + 1]:
                i += 1
                j += 1
            else:
                break

        self.top_pt = i, j
        self.features = {
            'center': segment_around_point(self.center_pt, self.blanks, length=3),
            'top': segment_around_point(self.top_pt, image > 20, length=5),
        }


class SevenFeatures:

    def __init__(self, image):
        digit_seg = image > 20
        self._skeleton = morphology.skeletonize(digit_seg)
        morphology.remove_small_objects(self._skeleton, min_size=8, connectivity=2, in_place=True)

        self.top_left = max_score_point(lambda i, j: -1 * (i ** 2 + j ** 2), self._skeleton)
        self.top_right = max_score_point(lambda i, j: -1 * (i ** 2 + (SIZE - 3 - j) ** 2), self._skeleton)
        self.bottom = max_score_point(lambda i, j: i, self._skeleton)
        g = np.where(self._skeleton, 1, 300)
        top, _ = graph.route_through_array(g, self.top_left, self.top_right, geometric=False)
        leg, _ = graph.route_through_array(g, self.top_right, self.bottom, geometric=False)
        self.features = {
            'top': segment_path(top, digit_seg, width=5),
            'leg': segment_path(leg, digit_seg, width=5),
        }


class NineFeatures:
    _kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [0, 0, 0]])

    def __init__(self, image):
        digit_seg = morphology.remove_small_objects(image > 120, min_size=8, connectivity=2)
        self._skeleton = morphology.skeletonize(digit_seg)
        self.circle_path = self._find_circle()
        circle = segment_path(self.circle_path, digit_seg)
        non_circle = np.logical_not(circle) & digit_seg
        leg = morphology.remove_small_objects(non_circle, min_size=8, connectivity=2)
        self.features = {
            'circle': circle,
            'leg': leg,
        }

    def _find_circle(self):
        i, j = max_score_point(lambda x, y: -1 * (x ** 2 + (SIZE / 2 - y) ** 2), self._skeleton)
        circle = [(i, j)]
        while True:
            if self._skeleton[i, j - 1]:
                j -= 1
                circle.append((i, j))
            elif self._skeleton[i + 1, j - 1]:
                i += 1
                j -= 1
                circle.append((i, j))
            else:
                break

        while not self._skeleton[i, j + 1]:
            if self._skeleton[i + 1, j + 1]:
                i += 1
                j += 1
                circle.append((i, j))
            elif self._skeleton[i + 1, j]:
                i += 1
                circle.append((i, j))
            elif self._skeleton[i + 1, j - 1]:
                i += 1
                j -= 1
                circle.append((i, j))
            else:
                break

        while True:
            if self._skeleton[i - 1, j + 1]:
                i -= 1
                j += 1
                circle.append((i, j))
            elif self._skeleton[i, j + 1]:
                j += 1
                circle.append((i, j))
            elif self._skeleton[i + 1, j + 1]:
                i += 1
                j += 1
                circle.append((i, j))
            else:
                break

        while True:
            if self._skeleton[i - 1, j - 1]:
                i -= 1
                j -= 1
                circle.append((i, j))
            elif self._skeleton[i - 1, j]:
                i -= 1
                circle.append((i, j))
            elif self._skeleton[i - 1, j + 1]:
                i -= 1
                j += 1
                circle.append((i, j))
            else:
                break

        while self._skeleton[i, j - 1]:
            j -= 1
            circle.append((i, j))

        i, j = circle[0]
        while True:
            if self._skeleton[i, j + 1]:
                j += 1
                circle.append((i, j))
            elif self._skeleton[i + 1, j + 1]:
                i += 1
                j += 1
                circle.append((i, j))
            elif self._skeleton[i - 1, j + 1]:
                i -= 1
                j += 1
                circle.append((i, j))
            else:
                break

        return set(circle)
