from glob import glob
import re
from . import basic_entities as entities


def compile_to_frame_obj(dictio, load_labels=True):
    frame = basic_entities.BasicFrame(dictio['png'])
    if load_labels:
        with open(dictio['csv']) as f:
            lines = [line.rstrip() for line in f]

        if len(lines) > 1:
            for line in lines[1:]:
                coords = line.split(',')
                frame.add_point((int(coords[0]), int(coords[1])))
    return frame


# Load Amsterdam Pedestrians into videos and frames
def load_videos(base_path, load_labels=True):
    png_regex = re.compile(r'(.*)\/([0-9]+)-([0-9]{10}).([0-9]+).png\Z')
    csv_regex = re.compile(r'(.*)\/([0-9]+)-([0-9]{10})([0-9]+)-tags.csv\Z')
    frame_dict = {}
    for file in glob('{}*'.format(base_path)):
        png_match = png_regex.match(file)
        csv_match = csv_regex.match(file)
        if csv_match is not None:
            frame_number = int(csv_match.group(2))
            uid = int(csv_match.group(3))
        elif png_match is not None:
            frame_number = int(png_match.group(2))
            uid = int(png_match.group(3))
        else:
            print("Nothing!!! Something went wrong!!")
            print(file)
            break

        if frame_number not in frame_dict:
            frame_dict[frame_number] = {}

        if uid not in frame_dict[frame_number]:
            frame_dict[frame_number][uid] = {}

        if csv_match is not None:
            frame_dict[frame_number][uid]['csv'] = file
        elif png_match is not None:
            frame_dict[frame_number][uid]['png'] = file

    frame_numbers = sorted(list(frame_dict.keys()))
    videos = []
    for begin_frame in frame_dict[0]:
        video = basic_entities.BasicVideo(
            png_regex.match(frame_dict[0][begin_frame]['png']).group(1),
            load_labels
        )

        for frame_num in frame_numbers:
            added = False
            for i in range(-20, 2):
                if begin_frame + frame_num + i in frame_dict[frame_num]:
                    added = True
                    video.add_frame(compile_to_frame_obj(
                        frame_dict[frame_num][begin_frame + frame_num + i],
                        load_labels
                    ))
                    break

            if added == False:
                break
        videos.append(video)

    return videos
