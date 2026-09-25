import unittest

from PIL import Image

from training_pipeline.mask_utils import resize_binary_mask


class ResizeBinaryMaskTests(unittest.TestCase):
    def test_resize_preserves_binary_labels_and_boundary(self):
        mask = Image.new("L", (2, 2), color=0)
        mask.putpixel((0, 0), 255)

        resized = resize_binary_mask(mask, (4, 4))

        self.assertEqual(resized.size, (4, 4))
        self.assertEqual(set(resized.getdata()), {0, 255})
        self.assertEqual(sum(value == 255 for value in resized.getdata()), 4)

    def test_size_is_height_then_width(self):
        mask = Image.new("L", (2, 3), color=255)

        self.assertEqual(resize_binary_mask(mask, (6, 4)).size, (4, 6))


if __name__ == "__main__":
    unittest.main()
