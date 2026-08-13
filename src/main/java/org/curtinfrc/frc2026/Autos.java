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
   * corresponding command. "Start Intake" runs until "Stop Intake", and "Shoot" meters three
   * sensorless shots while waiting for flywheel recovery between each one.
   */
  private static void configureEventBindings(AutoFactory factory, Drive drive, Shooter shooter) {
    factory
        .bind("Start Intake", shooter.intake())
        .bind("Stop Intake", shooter.stopOnce())
        .bind(
            "Shoot",
            shooter.shootBurst(
                drive::getDistanceToShotTarget, drive::isReturningFuelToAllianceSide, 3))
        // Keep the existing lowercase marker names usable in previously-authored paths.
        .bind("intake", shooter.intake())
        .bind("stop", shooter.stopOnce())
        .bind(
            "shoot",
            shooter.shootBurst(
                drive::getDistanceToShotTarget, drive::isReturningFuelToAllianceSide, 3))
        .bind(
            "score",
            shooter.shootBurst(
                drive::getDistanceToShotTarget, drive::isReturningFuelToAllianceSide, 1))
        .bind(
            "shoot2",
            shooter.shootBurst(
                drive::getDistanceToShotTarget, drive::isReturningFuelToAllianceSide, 2))
        .bind(
            "shoot3",
            shooter.shootBurst(
                drive::getDistanceToShotTarget, drive::isReturningFuelToAllianceSide, 3));
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
