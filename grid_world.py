#lab code
import numpy as np
from matplotlib import pyplot as plt
from perlin_noise import PerlinNoise
from parameters import vals

SIZE = vals.grid_size
PAD_WIDTH = vals.pad_width
N = 5

#lab code
def bin_topology(grid, height_bins=N):
    minimum = grid.min()
    maximum = grid.max()
    domain = maximum - minimum

    grid = (grid - minimum) * 5 / domain

    grid_digitized = np.digitize(grid, range(height_bins + 1))

    return grid_digitized

#lab code
def get_grid(size=SIZE, padding=PAD_WIDTH, n=N):
    noise1 = PerlinNoise(octaves=3)
    noise2 = PerlinNoise(octaves=6)
    noise3 = PerlinNoise(octaves=12)
    noise4 = PerlinNoise(octaves=24)

    xpix, ypix = size, size
    pic = []
    for i in range(xpix):
        row = []
        for j in range(ypix):
            noise_val = noise1([i / xpix, j / ypix])
            noise_val += 0.5 * noise2([i / xpix, j / ypix])
            noise_val += 0.25 * noise3([i / xpix, j / ypix])
            noise_val += 0.125 * noise4([i / xpix, j / ypix])

            row.append(noise_val)
        pic.append(row)

    grid = np.array(pic)
    grid = bin_topology(grid, n)
    mined_grid = assign_mines(grid)

    grid = np.pad(mined_grid, padding, mode='constant', constant_values=-1).astype(int)

    return grid

#lab code
def assign_mines(grid, num_of_mines=vals.num_mines):
    mine_rows = np.random.randint(vals.grid_size, size=(1, num_of_mines))
    mine_cols = np.random.randint(vals.grid_size, size=(1, num_of_mines))

    rc = np.concatenate([mine_rows, mine_cols], axis=0)
    rc = np.unique(rc, axis=-1)
    print(rc)

    grid[rc[0, :], rc[1, :]] = vals.mine_value

    return grid

#lab code
if __name__ == "__main__":
    g = get_grid()
    print(g)
    print("Grid Shape set to {}x{}, with a padding width {}".format(g.shape[0], g.shape[1], PAD_WIDTH))
    mine_locations = np.asarray(g == vals.mine_value).nonzero()
    print(mine_locations)
    num_of_mines = mine_locations[0].shape[0]
    abs_loc = np.array(mine_locations) - np.tile(np.array(vals.starting_pos), (num_of_mines, 1)).T

    print(abs_loc)
    fig = plt.figure(dpi=150, figsize=(4, 4))
    plt.imshow(g, cmap='gray')
    fig.savefig("perlin_noise_binned_padded.png")
    plt.show()
