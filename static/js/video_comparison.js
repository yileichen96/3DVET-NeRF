// Written by Dor Verbin, October 2021
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    var position = 0.5;
    var vidWidth = vid.videoWidth / 2;
    var vidHeight = vid.videoHeight;
    var mergeContext = videoMerge.getContext("2d");

    // 设置默认播放状态为 true，视频初始自动播放
    let isPlaying = true;
    let animationFrameId = null;

    // 检测 videoMerge 是否在视口内的回调函数
    function handleVisibilityChange(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                if (!animationFrameId) {
                    animationFrameId = requestAnimationFrame(drawLoop);
                }
            } else {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        });
    }

    // 设置 Intersection Observer
    let observer = new IntersectionObserver(handleVisibilityChange, { threshold: 0.1 });
    observer.observe(videoMerge);
    
    if (vid.readyState > 3) {
        // 启动视频播放
        vid.play();
        
        function drawLoop() {
            // 每一帧绘制 canvas
            mergeContext.clearRect(0, 0, videoMerge.width, videoMerge.height);

            if (isPlaying) {
                mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
                var colStart = (vidWidth * position).clamp(0.0, vidWidth);
                var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);
                mergeContext.drawImage(vid, colStart + vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
            } else {
                // 暂停时保持最后一帧（不重复设置 currentTime）
                mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
                var colStart = (vidWidth * position).clamp(0.0, vidWidth);
                var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);
                mergeContext.drawImage(vid, colStart + vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
            }
            
            // 绘制箭头及边框
            var arrowLength = 0.09 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 10;
            var arrowWidth = 0.007 * vidHeight;
            var currX = vidWidth * position;

            mergeContext.beginPath();
            mergeContext.arc(currX, arrowPosY, arrowLength * 0.7, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#FFD79340";
            mergeContext.fill();

            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth * position, 0);
            mergeContext.lineTo(vidWidth * position, vidHeight);
            mergeContext.closePath();
            mergeContext.strokeStyle = "#444444";
            mergeContext.lineWidth = 5;
            mergeContext.stroke();

            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2, arrowPosY);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2, arrowPosY);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowWidth / 2);
            mergeContext.lineTo(currX, arrowPosY - arrowWidth / 2);
            mergeContext.closePath();
            mergeContext.fillStyle = "#444444";
            mergeContext.fill();
            
            animationFrameId = requestAnimationFrame(drawLoop);
        }
        
        // 切换播放状态函数
        function togglePlay() {
            isPlaying = !isPlaying;
            if (isPlaying) {
                vid.play();
            } else {
                vid.pause();
            }
        }

        // 添加点击和触摸事件监听
        videoMerge.addEventListener("click", togglePlay);
        videoMerge.addEventListener("touchend", togglePlay);

        function trackLocation(e) {
            let bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
        }
        function trackLocationTouch(e) {
            let bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - bcr.x) / bcr.width);
        }

        videoMerge.addEventListener("mousemove", trackLocation, false); 
        videoMerge.addEventListener("touchstart", trackLocationTouch, false);
        videoMerge.addEventListener("touchmove", trackLocationTouch, false);
    } 
}

// 扩展 Number 原型函数
Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};

function resizeAndPlay(element) {
  var cv = document.getElementById(element.id + "Merge");
  cv.width = element.videoWidth / 2;
  cv.height = element.videoHeight;
  // 确保视频默认播放
  element.play();
  element.style.height = "0px";  // 隐藏 video 元素但继续播放供 canvas 绘制
  playVids(element.id);
}
