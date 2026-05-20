import xos


class GameOfLife(xos.Application):
    headless: bool = False
    device: None | str = None  # auto

    def __init__(self):
        super().__init__()

        # 3×3 Life kernel (no center)
        self.kernel = xos.tensor(
            [[1, 1, 1],
             [1, 0, 1],
             [1, 1, 1]],
            device="gpu",
            dtype=xos.uint8
        )

        self.randomize_state()
        self.update_framebuffer()

    def randomize_state(self):
        h, w, _ = self.frame.tensor.shape
        device = self.frame.tensor.device

        # allocate resolution-matching simulation buffers (0/1)
        self.state = xos.zeros((h, w), dtype=xos.uint8, device=device)
        self.next_state = xos.zeros_like(self.state)

        xos.random.uniform_fill(self.state, 0.0, 1.0)
        self.state = (self.state > 0.5).to(xos.uint8)

    def update_framebuffer(self):
        # Expand 0/1 → 0/255 RGB for display
        rgb = (self.state * 255).unsqueeze(-1).repeat(3, axis=-1)
        self.frame.tensor[:] = rgb

    def tick(self):
        N = xos.ops.convolve(
            self.state,
            self.kernel,
            inplace=False,
            padding="same"
        )

        # # Conway rule (vector form)
        # self.next_state = ((N == 3) | ((self.state == 1) & (N == 2))).to(xos.uint8)
        # print(self.next_state)

        # # swap buffers
        # self.state, self.next_state = self.next_state, self.state

        # self.update_framebuffer()
        pass

    # ------------------------------------------------------------------ resize

    def on_screen_size_change(self, width, height):
        self.randomize_state()
        self.update_framebuffer()


if __name__ == "__main__":
    app = GameOfLife()
    app.verbosities.function_calls = True
    app.run()