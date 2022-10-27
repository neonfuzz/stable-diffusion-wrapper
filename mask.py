"""Tools to draw custom masks for StableDiffusion infilling.

Classes:
    StableMasker - interactively draw an infill mask
"""

from copy import copy

from PIL import Image
import pygame
import pygame.locals as pgl


def _pil_to_pygame(img: Image.Image):
    str_format = "RGB"
    bytes_str = img.tobytes("raw", str_format)
    surface = pygame.image.fromstring(
        bytes_str, img.size, str_format
    ).convert()
    return surface


class StableMasker:
    """Create masks for infill with Stable Diffusion.

    Instance Attributes:
        thickness (int): size of brush, can be modified with the scrollwheel
        width (int): width of image, default=512, set when called
        height (int): height of image, default=512, set when called
        mask (Image.Image): created mask, default all-white

    Methods:
        quit: exit the pygame program loop
        change_rad: update the `thickness`
        draw: draw on image / update mask
        reset: start over
    """

    def __init__(self, thickness: int = 30):
        """Initialize.

        Args:
            thickness (int): starting cursor thickness
        """
        self.width = 512
        self.height = 512
        self.thickness = thickness
        self.mask = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        self._image = None
        self._screen = None
        self._mask = None

    def __call__(self, img: Image.Image):
        """Create a mask on an image.

        Args:
            img (Image.Image): image to mask.

        When Called:
            Displays `img` to screen in an editable window.
            Left click adds mask; right click erases mask.
            Scroll up/down changes brush size.
            "Escape" or "Return" exit and return the mask.
            "R" erases all mask and starts over.
        """
        self.width, self.height = img.size
        pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        self._image = _pil_to_pygame(img)
        self._screen.blit(self._image, (0, 0))
        pygame.display.update()

        self._mask = copy(self._image)
        self._mask.fill((0, 0, 0))
        self._mask.set_colorkey((0, 0, 0))  # Black is alpha.

        running = True
        while running:
            for event in pygame.event.get():
                running = self._handle_pygame_events(event)
        self.quit()
        return self.mask

    def _handle_pygame_events(self, event):
        if event.type == pgl.QUIT or (
            event.type == pgl.KEYDOWN
            and event.key in (pgl.K_ESCAPE, pgl.K_RETURN)
        ):
            return False

        if event.type == pgl.KEYDOWN and event.key == pgl.K_r:
            self.reset()
        elif event.type == pgl.MOUSEWHEEL:
            self.change_rad(amt=2 * event.y)
        elif pygame.mouse.get_pressed() == (1, 0, 0):
            # Left click.
            self.draw()
        elif pygame.mouse.get_pressed() == (0, 0, 1):
            # Right click.
            self.draw((0, 0, 0))

        return True

    def quit(self):
        """Quit out of the pygame environment and calculate mask."""
        str_format = "RGB"
        mask_bytes = pygame.image.tostring(self._mask, str_format, False)
        self.mask = Image.frombytes(
            str_format, (self.width, self.height), mask_bytes
        )
        self._image = None
        self._screen = None
        self._mask = None
        pygame.quit()

    def change_rad(self, amt: int):
        """Update `thickness` by `amt`."""
        self.thickness += amt
        self.thickness = max(1, self.thickness)
        self.thickness = min(max(self.width, self.height), self.thickness)

    def draw(self, color: tuple = None):
        """Draw on screen.

        Args:
            color (tuple): RGB color, default=(255, 255, 255)
        """
        color = color or (255, 255, 255)
        pygame.draw.circle(
            surface=self._mask,
            color=color,
            center=pygame.mouse.get_pos(),
            radius=self.thickness,
            width=0,
        )
        self._screen.blit(self._image, (0, 0))
        self._screen.blit(self._mask, (0, 0))
        pygame.display.update()

    def reset(self):
        """Reset screen to base image."""
        self._mask.fill((0, 0, 0))
        self._screen.blit(self._image, (0, 0))
        pygame.display.update()
