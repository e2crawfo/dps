import dominate
import matplotlib.pyplot as plt
from matplotlib import animation
from dominate import tags
from scipy.misc import imsave as sp_imsave
from skimage import img_as_int
from bs4 import BeautifulSoup
import os
from io import BytesIO
from base64 import b64encode, b64decode
import datetime
import matplotlib.image as mpimg
import argparse


def format_dict(d):
    s = ['']

    def helper(d, s, depth=0):
        for k, v in sorted(d.items(), key=lambda x: x[0]):
            if isinstance(v, dict):
                s[0] += ("  ")*depth + ("%s: {" % k) + ',\n'
                helper(v, s, depth+1)
                s[0] += ("  ")*depth + ("}") + ',\n'
            else:
                s[0] += ("  ")*depth + "%s: %s" % (k, v) + ',\n'

    helper(d, s)
    return s[0]


class HTMLReport(object):
    def __init__(self, path, images_per_row=1000, default_image_width=400):
        self.path = path
        title = datetime.datetime.today().strftime(
            "Report %Y-%m-%d_%H-%M-%S_{}".format(os.uname()[1])
        )
        self.doc = dominate.document(title=title)
        self.images_per_row = images_per_row
        self.default_image_width = default_image_width
        self.t = None
        self.row_image_count = 0

    def add_header(self, str):
        with self.doc:
            tags.h3(str, style='word-wrap: break-word; white-space: pre-wrap;')
        self.t = None
        self.row_image_count = 0

    def add_text(self, str):
        with self.doc:
            tags.p(str, style='word-wrap: break-word; white-space: pre-wrap;')
        self.t = None
        self.row_image_count = 0

    def _add_table(self, border=1):
        self.row_image_count = 0
        self.t = tags.table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def _encode_image(self, img_arr):
        """Save the image array as PNG and then encode with base64 for embedding"""
        img_arr = img_as_int(img_arr)
        sio = BytesIO()
        sp_imsave(sio, img_arr, 'png')
        encoded = b64encode(sio.getvalue()).decode()
        sio.close()
        return encoded

    def add_image(self, im, title, txt='', width=None, font_pct=100):
        if width is None:
            width = self.default_image_width
        if self.t is None or self.row_image_count >= self.images_per_row:
            self._add_table()
        with self.t:
            # with tr():
            # with td(style="word-wrap: break-word;", halign="center", valign="top"):
            with tags.td(halign="center", valign="top"):
                with tags.p():
                    tags.img(
                        style="width:%dpx" % width,
                        src=r'data:image/png;base64,' + self._encode_image(im)
                    )
                    tags.br()
                    tags.p(
                        "{}\n{}".format(title, txt),
                        style='width:{}px; word-wrap: break-word; white-space: pre-wrap; font-size: {}%;'.format(
                            width,
                            font_pct
                        )
                    )
        self.row_image_count += 1

    def new_row(self):
        self.save()
        self.t = None
        self.row_image_count = 0

    def add_images(self, ims, titles, txts, width=256):
        for im, title, txt in zip(ims, titles, txts):
            self.add_image(im, title, txt, width)

    def save(self):
        with open(self.path, 'w') as f:
            f.write(self.doc.render())

    def __enter__(self):
        pass

    def __exit__(self, type_, value, tb):
        self.save()


def save_video(title, images, timesteps, output_dir):
    timesteps = timesteps or range(len(images))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    image = ax.imshow(images[0])
    timestep_text = ax.text(
        0.01, 0.01, '', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    def animate(t):
        plt.show()
        image.set_data(images[t])
        timestep_text.set_text("t={}".format(timesteps[t]))

    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=100)

    gif_filename = os.path.join(output_dir, title.replace(' ', '_') + '.gif')
    print("Saving video {} to {}.".format(title, gif_filename))
    anim.save(gif_filename, writer='imagemagick')

    plt.close(fig)


def report_to_videos(filename, output_dir, max_frames=None, take_every=None, first_row=0):

    os.makedirs(output_dir, exist_ok=True)

    with open(filename, 'r') as f:
        soup = BeautifulSoup(f)

        rows = soup.find_all('table')[first_row:]

        take_every = take_every or 1
        timesteps = range(0, len(rows), take_every)
        rows = [rows[t] for t in timesteps]
        max_frames = max_frames or len(rows)
        rows = rows[:max_frames]
        timesteps = timesteps[:max_frames]

        n_images = len(rows[0].find_all('td'))

        for i in range(n_images):
            title = None
            images = []
            for row in rows:
                td = row.find_all('td')[i]
                if title is None:
                    p = td.find_all('p')[1]
                    title = p.text.split('\n')[-1]

                img_tag = td.find_all('img')[0]
                _, data = img_tag.attrs['src'].split(',')
                io = BytesIO(b64decode(data))
                img = mpimg.imread(io, format='png')
                images.append(img)

            save_video(title, images, timesteps, output_dir)


def report_to_videos_cl():
    parser = argparse.ArgumentParser()
    parser.add_argument('report')
    parser.add_argument('output_dir')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--take-every', type=int, default=None)
    parser.add_argument('--first-row', type=int, default=None)
    args = parser.parse_args()

    report_to_videos(args.report, args.output_dir, args.max_frames, args.take_every, args.first_row)
