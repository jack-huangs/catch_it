import os

# Path to the 'rollouts' folder
video_dir = 'rollouts'

# Get a list of all .mp4 files in the directory (without manually listing them)
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

# Start HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Grid</title>
    <style>
        /* Ensures body and html take full height of the viewport */
        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
            background-color: white;
        }

        /* The grid container */
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 10px;
            padding: 10px;
            box-sizing: border-box;
        }

        /* Styling for the video element */
        video {
            width: 100%;
            height: auto;
            display: block;  /* Remove extra space below video */
            overflow: hidden; /* Ensures no scrollbars */
            -ms-overflow-style: none; /* Hide scrollbars for IE and Edge */
            scrollbar-width: none; /* Hide scrollbars for Firefox */
            border: none;
        }

        video::-webkit-scrollbar {
            display: none; /* Hide scrollbars for WebKit browsers */
        }
    </style>
</head>
<body>
    <div class="grid-container">
"""

# Add each video file to the grid (dynamically using the list of files)
for filename in video_files:
    video_path = os.path.join(video_dir, filename)
    html_content += f"""
        <video class="video" muted src="{video_path}"></video>
    """

# End the HTML content
html_content += """
    </div>

    <script>
        // Function to autoplay videos when they come into view
        function autoplayOnScroll() {
            const videos = document.querySelectorAll('.video');
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    const video = entry.target;
                    if (entry.isIntersecting) {
                        video.play();
                    } else {
                        video.pause();
                    }
                });
            }, { threshold: 0.5 }); // Play video when 50% visible

            videos.forEach(video => observer.observe(video));
        }

        // Start autoplay when page loads
        document.addEventListener('DOMContentLoaded', autoplayOnScroll);
    </script>
</body>
</html>
"""

# Write the HTML content to a file
output_html = 'video_grid.html'
with open(output_html, 'w') as f:
    f.write(html_content)

print(f"HTML file generated: {output_html}")

