from enum import Enum
from typing import List, Dict, Tuple, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.axes import Axes
from matplotlib.patches import Polygon


NUM_FACES = 8
NUM_FACES_PER_SQUARE = 4
NUM_TRIANGLES_PER_FACE = 9

SQUARE_SIZE = 6
GAP_SIZE = 1
FACE_SIZE = SQUARE_SIZE / 2
TRIANGLE_SIZE = FACE_SIZE / 3

# Constants for triangle indices
CORNER_INDICES = [0, 2, 6]
EDGE_INDICES = [1, 3, 4]
CENTER_INDICES = [5, 7, 8]
TRIANGLE_ROTATION_MAP = {0: 6, 1: 4, 2: 0, 3: 1, 4: 3, 5: 8, 6: 2, 7: 5, 8: 7}


# Face colors
class FaceColor(Enum):
    WHITE = "white"
    RED = "red"
    GREEN = "green"
    PURPLE = "purple"
    YELLOW = "yellow"
    GRAY = "gray"
    BLUE = "blue"
    ORANGE = "orange"
    VOID = PINK = "pink"  # special color for testing -- not a side color


class FaceNotation(Enum):
    """
    Face notations:
    - "F" represents turning the front face clockwise
    - "F'" represents turning the front face counterclockwise
    etc.
    """

    FRONT = "F"
    BACK = "B"
    LEFT = "L"
    RIGHT = "R"
    UP = "U"
    DOWN = "D"
    BACKLEFT = "BL"
    BACKRIGHT = "BR"


class FaceTurningOctahedron:
    def __init__(self) -> None:
        """Faces are numbered as follows

           1          5
        0     2    4     6
           3          7

        The triangles on each face are numbered as follows:
        - Corners: 0, 2, 6
        - Edges: 1, 3, 4
        - Centers: 5, 7, 8
        Each sequence starts at the tip of the face facing towards the intersection of the square and continues clockwise
        """
        # Initialize the puzzle with solved state
        self.faces: List[List[FaceColor]] = [
            # These are Square 0
            [FaceColor.PINK] + [FaceColor.WHITE] * (NUM_TRIANGLES_PER_FACE - 1),  # White Face
            [FaceColor.RED] * NUM_TRIANGLES_PER_FACE,  # Red face
            [FaceColor.GREEN] * NUM_TRIANGLES_PER_FACE,  # Blue face
            [FaceColor.PURPLE] * NUM_TRIANGLES_PER_FACE,  # Green face
            # These are Square 1
            [FaceColor.YELLOW] * NUM_TRIANGLES_PER_FACE,  # Yellow face
            [FaceColor.GRAY] * NUM_TRIANGLES_PER_FACE,  # Orange face
            [FaceColor.BLUE] * NUM_TRIANGLES_PER_FACE,  # Purple face
            [FaceColor.ORANGE] * NUM_TRIANGLES_PER_FACE,  # Gray face
        ]

        # self.faces = [
        #     [x for x in FaceColor]
        #     for _ in range(8)
        # ]

    def reset(self) -> None:
        self.__init__()

    def colors_by_triangle(self) -> Dict[Tuple[int, int, int], str]:
        squares = 2
        faces_per_square = 4
        triangles_per_face = 9
        return {
            (idx, jdx, kdx): self.faces[idx * faces_per_square + jdx][kdx].value
            for idx in range(squares)
            for jdx in range(faces_per_square)
            for kdx in range(triangles_per_face)
        }

    def rotate_face(self, face_index: int, counter_clockwise: bool = False) -> None:
        self.rotate_face_clockwise(face_index)
        if counter_clockwise:
            self.rotate_face_clockwise(face_index)

    def rotate_face_clockwise(self, face_index: int) -> None:
        # Rotate triangles on the face
        self.faces[face_index] = self.rotate_triangle_clockwise(self.faces[face_index])

        # Get the initial state of the faces
        init_faces = [face[:] for face in self.faces]

        # Move the side pieces as part of the layer rotation
        adjacent_side_faces = self.get_adjacent_side_faces(face_index)
        for prev_idx, next_idx in zip(adjacent_side_faces, adjacent_side_faces[1:] + adjacent_side_faces[:1]):
            sigil = (face_index % 4) - (prev_idx % 4)
            # NOTE: This might not ever happen... can possibly be removed
            if sigil == 0:
                sigil = -2
            indices = self.get_side_layer_indices_by_sigil(sigil)

            for i in indices:
                prev_sub_idx = i
                next_sub_idx = TRIANGLE_ROTATION_MAP[i]

                if i in EDGE_INDICES:
                    rev_dict = {v: k for k, v in TRIANGLE_ROTATION_MAP.items()}
                    next_sub_idx = rev_dict[prev_sub_idx]

                self.faces[next_idx][next_sub_idx] = init_faces[prev_idx][prev_sub_idx]

        # Move the corner pieces not moved as part of the side layer rotation
        adjacent_corner_faces = self.get_adjacent_corner_faces(face_index)
        for prev_idx, next_idx, prev_triangle_idx, next_triangle_idx in zip(
            adjacent_corner_faces,
            adjacent_corner_faces[1:] + adjacent_corner_faces[:1],
            CORNER_INDICES,
            CORNER_INDICES[1:] + CORNER_INDICES[:1],
        ):
            self.faces[next_idx][next_triangle_idx] = init_faces[prev_idx][prev_triangle_idx]

    def get_adjacent_side_faces(self, face_index: int) -> List[int]:
        # Which is first doesn't matter, but should reflect clockwise rotation of key face
        return {
            0: [6, 1, 3],
            1: [0, 5, 2],
            2: [1, 4, 3],
            3: [0, 2, 7],
            4: [2, 5, 7],
            5: [6, 4, 1],
            6: [0, 7, 5],
            7: [4, 6, 3],
        }[face_index]

    def get_adjacent_corner_faces(self, face_index: int) -> List[int]:
        # The first item in the list should be the one opposite the key face
        return {
            0: [2, 7, 5],
            1: [3, 6, 4],
            2: [0, 5, 7],
            3: [1, 4, 6],
            4: [6, 3, 1],
            5: [7, 2, 0],
            6: [4, 1, 3],
            7: [5, 0, 2],
        }[face_index]

    def get_side_layer_indices_by_sigil(self, sigil: int) -> List[int]:
        return {
            -1: [0, 1, 2, 5, 7],
            3: [0, 1, 2, 5, 7],
            -2: [2, 4, 6, 7, 8],
            2: [2, 4, 6, 7, 8],
            -3: [0, 3, 6, 5, 8],
            1: [0, 3, 6, 5, 8],
        }[sigil]

    def rotate_triangle_clockwise(self, face: List[FaceColor]) -> List[FaceColor]:
        return [face[TRIANGLE_ROTATION_MAP[i]] for i in range(NUM_TRIANGLES_PER_FACE)]


def draw_puzzle(ax: Axes, fto: FaceTurningOctahedron) -> None:
    collection = draw_fto_net_with_colors(fto.colors_by_triangle())
    ax.add_collection(collection)
    ax.autoscale_view()


def update_plot(fto: FaceTurningOctahedron, ax: Axes) -> None:
    ax.clear()
    ax.set_title("Face Turning Octahedron")
    draw_puzzle(ax, fto)


# This is a little hacky workaround to deal with some repeated frame idxs bug from matplotlib
prev_i = None


def animate(i: int, fto: FaceTurningOctahedron, ax: Axes) -> None:
    global prev_i
    if i == prev_i:
        return
    prev_i = i

    sigil = i % 4
    if sigil == 0:
        fto.rotate_face(2, counter_clockwise=True)
    elif sigil == 1:
        fto.rotate_face(0)
    elif sigil == 2:
        fto.rotate_face(2)
    else:
        fto.rotate_face(0, counter_clockwise=True)

    update_plot(fto, ax)


def draw_fto_net_with_colors(color_map: Optional[Dict[Tuple[int, int, int], str]] = None) -> PatchCollection:
    """
    Draws the FTO net with two squares, each containing 4 faces,
    with faces rotated so that their tips meet at the center.
    Each face is further divided into 9 triangles.

    :param color_map: Dictionary mapping triangle indices to colors.
                      The keys should be tuples like (square, face, triangle),
                      where:
                        - square: 0 (left) or 1 (right)
                        - face: 0-3 (within each square)
                        - triangle: 0-8 (within each face)
                      Example: {(0, 0, 0): 'red', (1, 2, 4): 'blue'}
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Define the size of each square and face

    # Define the two squares in the layout
    squares = [
        (0, 0),  # Left square
        (SQUARE_SIZE + GAP_SIZE, 0),  # Right square
    ]

    patches = []
    colors = []

    # Loop through each square
    for square_index, (square_x, square_y) in enumerate(squares):
        # Define the 4 main faces in each square
        CENTER = (square_x + FACE_SIZE, square_y + FACE_SIZE)
        face_vertices = [
            [
                CENTER,
                (square_x, square_y + SQUARE_SIZE),  # Top-left
                (square_x, square_y),
            ],  # Bottom-left
            [
                CENTER,
                (square_x + SQUARE_SIZE, square_y + SQUARE_SIZE),  # Top-right
                (square_x, square_y + SQUARE_SIZE),
            ],  # Top-left
            [
                CENTER,
                (square_x + SQUARE_SIZE, square_y),  # Bottom-right
                (square_x + SQUARE_SIZE, square_y + SQUARE_SIZE),
            ],  # Top-right
            [
                CENTER,
                (square_x, square_y),  # Bottom-left
                (square_x + SQUARE_SIZE, square_y),
            ],  # Bottom-right
        ]

        # Divide each face into 9 triangles
        for face_index, vtxs in enumerate(face_vertices):
            # Get the three vertices of the face
            v1, v2, v3 = vtxs

            # Subdivide the face into 9 triangles
            for i in range(3):
                for j in range(3 - i):
                    # Calculate the smaller triangle's vertices
                    t1 = (
                        v1[0] + i * (v2[0] - v1[0]) / 3 + j * (v3[0] - v1[0]) / 3,
                        v1[1] + i * (v2[1] - v1[1]) / 3 + j * (v3[1] - v1[1]) / 3,
                    )
                    t2 = (
                        t1[0] + (v2[0] - v1[0]) / 3,
                        t1[1] + (v2[1] - v1[1]) / 3,
                    )
                    t3 = (
                        t1[0] + (v3[0] - v1[0]) / 3,
                        t1[1] + (v3[1] - v1[1]) / 3,
                    )
                    patches.append(Polygon([t1, t2, t3], closed=True))

                    # Determine color from color_map if provided
                    color_key = (square_index, face_index, i * 3 + j)
                    # print(color_key)
                    # print(color_map)
                    colors.append(color_map.get(color_key, FaceColor.VOID.value) if color_map else FaceColor.VOID.value)

            # Add center pieces between triangles
            for i in range(3):
                for j in range(3 - i):
                    if i < 2 and j < 2 - i:
                        # Calculate center triangle vertices
                        center_t1 = (
                            v1[0] + (i + 1) * (v2[0] - v1[0]) / 3 + (j + 1) * (v3[0] - v1[0]) / 3,
                            v1[1] + (i + 1) * (v2[1] - v1[1]) / 3 + (j + 1) * (v3[1] - v1[1]) / 3,
                        )
                        center_t2 = (
                            center_t1[0] - (v2[0] - v1[0]) / 3,
                            center_t1[1] - (v2[1] - v1[1]) / 3,
                        )
                        center_t3 = (
                            center_t1[0] - (v3[0] - v1[0]) / 3,
                            center_t1[1] - (v3[1] - v1[1]) / 3,
                        )
                        patches.append(Polygon([center_t1, center_t2, center_t3], closed=True))

                        # Assign colors correctly for the center pieces
                        center_color_key = (
                            square_index,
                            face_index,
                            5 + j * 2 + i * 3,
                        )
                        # print("center i ", i, " j ", j)
                        # print("center", center_color_key)
                        colors.append(
                            color_map.get(center_color_key, FaceColor.VOID.value) if color_map else FaceColor.VOID.value
                        )

    return PatchCollection(patches, edgecolor="black", facecolor=colors, linewidths=1)


def main() -> None:
    # Example usage
    fto = FaceTurningOctahedron()
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, animate, frames=30, fargs=(fto, ax), interval=500)  # type: ignore
    plt.show()


if __name__ == "__main__":
    main()
