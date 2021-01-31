const express = require('express')
const bodyParser = require('body-parser');
const app = express()
const port = 3000

var glob = require("glob")
var fs = require('fs');

const base_path = '../videos/'

let videos = null

let write_labeled_data = (video_data, label_data) => {
    let data = {
        'video': video_data,
        'labels': label_data
    }
    json = JSON.stringify(data)

    let name = video_data['name']
    console.log(`${base_path}${name}info.json`)
    console.log(data)
    fs.writeFile(`${base_path}${name}info.json`, json, 'utf8', () => {});
    return;
}

let load_all_videos = () => {
    glob(`${base_path}*/`, (er, files) => {
        console.log(`${base_path}*/`)
        videos = []
        files.forEach(file => {
            console.log(`Check ${file}`)
            if(!fs.existsSync(`${file}info.json`)) {
                videos.push(file.slice(base_path.length))
            } else {
                console.log(`${file} is labeled already`)
            }
        })

        console.log(`Total labelers left: ${videos.length}`)
    })
}

load_all_videos()


app.use(bodyParser.urlencoded({ extended: true }));
app.use('/data', express.static(base_path))
app.use('/', express.static('public'))

app.get('/next', (req, res) => {
    if(videos.length == 0) {
        res.send('')
    } else {
        video = videos.pop()
        console.log(`Next labeling: ${video}`)
        res.send(video)
    }
})

app.post('/send', (req, res) => {
    console.log('Submitted video')
    write_labeled_data(JSON.parse(req.body.video), JSON.parse(req.body.labels))
    res.send('success')
})

app.listen(port, () => {
  console.log(`Labeler app listening at http://localhost:${port}`)
})