# Nonverbal Behaviors Analysis

<table>
  <tr>
    <th rowspan="2">Modality</th>
    <th>Component</th>
    <th>Description</th>
    <th>Tool</th>
  </tr>
  <tr></tr>

  <tr>
    <td rowspan="2"><b>Face</b></td>
    <td>Facial Landmarks</td>
    <td>Facial Landmarks in 2D and 3D as normalized values</td>
    <td>OpenFace</td>
  </tr>
  <tr>
    <td>Facial Emotions</td>
    <td>Likelihood values of 7 distinct emotions</td>
    <td>RMN</td>
  </tr>

  <tr>
    <td rowspan="3"><b>Gaze (Normalized)</b></td>
    <td>Eye Gaze Landmarks</td>
    <td>Eye Gaze Landmarks in 2D and 3D</td>
    <td>OpenFace</td>
  </tr>
  <tr>
    <td>Eye Gaze Direction</td>
    <td>Eye gaze direction vector (x,y,z) and direction in radians (x,y)</td>
    <td>OpenFace</td>
  </tr>
  <tr>
    <td>Gaze Classification</td>
    <td>Gaze on Robot, Participant, Misc.</td>
    <td>OpenFace</td>
  </tr>

  <tr>
    <td rowspan="2"><b>Head</b></td>
    <td>Pose Estimation</td>
    <td>Location of the head with respect to camera (x,y,z)</td>
    <td>OpenFace</td>
  </tr>
  <tr>
    <td>Rotation</td>
    <td>Rotation is in radians around X,Y,Z axes (pitch, yaw, roll)</td>
    <td>OpenFace</td>
  </tr>

  <tr>
    <td rowspan="1"><b>Body</b></td>
    <td>Pose Landmarks</td>
    <td>Body Landmarks in 2D and 3D as normalized and world coordinate values</td>
    <td>MediaPipe</td>
  </tr>
</table>