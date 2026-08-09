package org.curtinfrc.frc2026;

import choreo.auto.AutoChooser;
import choreo.auto.AutoFactory;
import edu.wpi.first.wpilibj.Alert;
import edu.wpi.first.wpilibj.Alert.AlertType;
import edu.wpi.first.wpilibj.Filesystem;
import edu.wpi.first.wpilibj2.command.Commands;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.curtinfrc.frc2026.drive.Drive;
import org.curtinfrc.frc2026.shooter.Shooter;

/** Configures Choreo trajectory following and dashboard autonomous selection. */
public final class Autos {
  private static final String TRAJECTORY_EXTENSION = ".traj";
  private static final double AUTO_INTAKE_SECONDS = 2.0;

  private final AutoChooser chooser = new AutoChooser("Do Nothing");
  private final Alert noTrajectoriesAlert =
      new Alert(
          "No Choreo trajectories found in deploy/choreo; only Do Nothing is available.",
          AlertType.kWarning);

  public Autos(Drive drive, Shooter shooter) {
    AutoFactory factory =
        new AutoFactory(drive::getPose, drive::setPose, drive::followTrajectory, true, drive);

    configureEventBindings(factory, drive, shooter);

    List<String> trajectoryNames = findDeployedTrajectories();
    noTrajectoriesAlert.set(trajectoryNames.isEmpty());
    for (String trajectoryName : trajectoryNames) {
      chooser.addCmd(
          trajectoryName,
          () ->
              Commands.sequence(
                      factory.resetOdometry(trajectoryName), factory.trajectoryCmd(trajectoryName))
                  .withName("Choreo: " + trajectoryName));
    }
  }

  public AutoChooser getChooser() {
    return chooser;
  }

  /**
   * Global Choreo event-marker names. Add one of these markers to any trajectory to run the
   * corresponding command. Use shoot or score for one feeder pulse, or shoot2/shoot3 for a
   * sensorless two- or three-pulse burst. Every shot waits for flywheel recovery.
   */
  private static void configureEventBindings(AutoFactory factory, Drive drive, Shooter shooter) {
    factory
        .bind("intake", shooter.intake().withTimeout(AUTO_INTAKE_SECONDS))
        .bind("shoot", shooter.shootBurst(drive::getDistanceToAllianceHub, 1))
        .bind("score", shooter.shootBurst(drive::getDistanceToAllianceHub, 1))
        .bind("shoot2", shooter.shootBurst(drive::getDistanceToAllianceHub, 2))
        .bind("shoot3", shooter.shootBurst(drive::getDistanceToAllianceHub, 3))
        .bind("stop", shooter.stop());
  }

  private static List<String> findDeployedTrajectories() {
    Path choreoDirectory = Filesystem.getDeployDirectory().toPath().resolve("choreo");
    if (!Files.isDirectory(choreoDirectory)) {
      return List.of();
    }

    try (var files = Files.list(choreoDirectory)) {
      return files
          .filter(Files::isRegularFile)
          .map(path -> path.getFileName().toString())
          .filter(name -> name.endsWith(TRAJECTORY_EXTENSION))
          .map(name -> name.substring(0, name.length() - TRAJECTORY_EXTENSION.length()))
          .sorted()
          .toList();
    } catch (IOException exception) {
      return List.of();
    }
  }
}
