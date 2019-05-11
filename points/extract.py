import numpy as np
import operator

size = 28


def max_score_point(score, mask):
    score_map = np.fromfunction(score, mask.shape) * mask
    index = np.argmax(score_map)
    index = np.unravel_index(index, mask.shape)
    return index


class FourInterestPoints:
    def __init__(self, skeleton):
        self._skeleton = skeleton
        self._find_points()

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

    def _find_points(self):
        points = np.argwhere(self._skeleton)
        self.bottom = max(points, key=operator.itemgetter(0, 1))

        self.top_right = max_score_point(lambda i, j: j + size - i, self._skeleton)

        top_left = max_score_point(lambda i, j: size - j + size - i, self._skeleton)
        self.top_left = self.climb_up(top_left)

        middle_left = max_score_point(lambda i, j: size / 2 - 0.5 * abs(size / 2 - i) + size - j, self._skeleton)
        self.middle_left = self.climb_down(middle_left)

        has_left_neighbor = np.roll(self._skeleton, 1, axis=1)
        has_left_neighbor += np.roll(self._skeleton, 2, axis=1)
        has_left_neighbor += np.roll(self._skeleton, 3, axis=1)
        self.middle_right = max_score_point(lambda i, j: j - abs(i - middle_left[0]),
                                            has_left_neighbor * self._skeleton)
