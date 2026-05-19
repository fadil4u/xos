import xos


class TVApp(xos.Application):
    headless: bool = False
    # device: None = auto (GPU on native, CPU on wasm), "cpu" / "gpu" to force
    device: None | str = None # "gpu"

    def __init__(self):
        super().__init__()

        # frame is initialized with random static
        self.randomize_frame()
        self.randomize_kernel()

    def randomize_frame(self):
        xos.random.uniform_fill(self.frame.tensor, 0.0, 1.0)
        self.binarize_frame()

    def binarize_frame(self):
        mask = self.frame.tensor > 0.5
        self.frame.tensor[mask] = 255
        self.frame.tensor[~mask] = 0

    def randomize_kernel(self):
        # self.kernel = xos.random.uniform(0.001, 1.001, shape=(3, 3, 3), dtype=xos.float32)
        self.kernel = xos.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    def tick(self):
        # convolution tv will convolve the random frame
        neighbor_counts = xos.ops.convolve(self.frame.tensor, self.kernel, inplace=False, padding="same").to(xos.uint8)

        state = self.frame.tensor

        one_cells = state == 1
        zero_cells = state == 0

        # product rules
        state[one_cells and neighbor_counts < 2] = 0                                # only if the original cell was 1
        state[one_cells and neighbor_counts == 2 or neighbor_counts == 3] = 1       # only if the original cell was 1
        state[one_cells and neighbor_counts > 3] = 0                                # only if the original cell was 1
        state[zero_cells and neighbor_counts == 3] = 1                              # only if the original cell was 0

        self.binarize_frame()
        print(self.frame.tensor.device, self.frame.tensor.shape, self.fps)

    def on_screen_size_change(self, width, height):
        self.randomize_frame()
        self.randomize_kernel()
        print(width, height)

    def on_events(self):
        pass


if __name__ == "__main__":
    TVApp().run()