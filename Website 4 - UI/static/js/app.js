let canvas = document.querySelector("#canvas");
let context = canvas.getContext("2d");
let video = document.querySelector("#video");

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
    video.srcObject = stream;
    video.play();
  });
}

document.getElementById("snap").addEventListener("click", () => {
  context.drawImage(video, 0, 0, 640, 480);
  let dataURL = canvas.toDataURL(); // convert canvas image to data URL
  localStorage.setItem("capturedImage", dataURL); // store data URL in local storage
});


function dataURLToBlob(dataURL) {
  const arr = dataURL.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
}

function snap() {
  if (!video) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  var dataURL = canvas.toDataURL('image/png');

  var xhr = new XMLHttpRequest();
  xhr.open('POST', 'http://127.0.0.1:5000/upload', true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onreadystatechange = function () {
    if (xhr.readyState == 4) {
      if (xhr.status == 200) {
        var response = JSON.parse(xhr.responseText);
        var message = response.message;
        var gender = response.gender;
        var emotion = response.emotion;
        console.log(message);
        console.log(gender);
        console.log(emotion);

        window.location.href = '/resultpage?gender=' + encodeURIComponent(gender) + '&emotion=' + encodeURIComponent(emotion);
      } else {
        console.error('Failed to get results. Status code:', xhr.status);
      }
    }
  };
  xhr.send(JSON.stringify({
     photo: dataURL
  }))
};


function recommendation(gender, emotion){
  var animePlaylist = '';

  if(emotion == 'Angry'){
    if(gender == 'Male'){
      animePlaylist = 'https://myanimelist.net/anime/genre/36/Slice_of_Life';
    }
    else if(gender == 'Female'){
      animePlaylist = 'https://myanimelist.net/anime/genre/1/Action';
    }
  }
  else if(emotion == 'Happy'){
    if(gender == 'Male'){
      animePlaylist = 'https://myanimelist.net/anime/genre/14/Horror';
    }
    else if(gender == 'Female'){
      animePlaylist = 'https://myanimelist.net/anime/genre/8/Romance';
    }
  }
  else if(emotion == 'Neutral'){
    if(gender == 'Male'){
      animePlaylist = 'https://myanimelist.net/anime/genre/10/Fantasy';
    }
    else if(gender == 'Female'){
      animePlaylist = 'https://myanimelist.net/anime/genre/13/Mystery';
    }
  }
  else if(emotion == 'Sad'){
    if(gender == 'Male'){
      animePlaylist = 'https://myanimelist.net/anime/genre/4/Comedy';
    }
    else if(gender == 'Female'){
      animePlaylist = 'https://myanimelist.net/anime/genre/15/School';
    }
  }
  else if(emotion == 'Surprise'){
    if(gender == 'Male'){
      animePlaylist = 'https://myanimelist.net/anime/genre/2/Adventure';
    }
    else if(gender == 'Female'){
      animePlaylist = 'https://myanimelist.net/anime/genre/19/Shoujo';
    }
  }

  document.getElementById("emotion").innerHTML = emotion;
  document.getElementById("gender").innerHTML = gender;
  window.open(animePlaylist, '_blank');
}

  
  
  
  
  
  
  