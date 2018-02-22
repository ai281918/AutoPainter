import Painter

def celebA_128(use_model, rgb, reconstruct_iter, update_freq, input_size, show_area):
    if use_model:
        from painter_celebA_128 import VAEGAN_128_RGB as model

    p = Painter.Painter(use_model, rgb, reconstruct_iter, update_freq, input_size, show_area, model)
    p.main()


def main():
    celebA_128(True, True, 3, 5, (128, 128), (256, 256, 3))

if __name__ == '__main__':
    main()