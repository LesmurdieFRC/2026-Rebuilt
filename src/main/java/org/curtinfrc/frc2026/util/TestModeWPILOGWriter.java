package org.curtinfrc.frc2026.util;

import org.littletonrobotics.junction.LogDataReceiver;
import org.littletonrobotics.junction.LogTable;
import org.littletonrobotics.junction.wpilog.WPILOGWriter;

/** Opens an AdvantageKit WPILOG only while the robot is enabled in Test mode. */
public final class TestModeWPILOGWriter implements LogDataReceiver {
  private final String logDirectory;
  private WPILOGWriter writer;

  public TestModeWPILOGWriter(String logDirectory) {
    this.logDirectory = logDirectory;
  }

  @Override
  public void putTable(LogTable table) throws InterruptedException {
    boolean testEnabled =
        table.get("DriverStation/Test", false) && table.get("DriverStation/Enabled", false);

    if (testEnabled) {
      if (writer == null) {
        writer = new WPILOGWriter(logDirectory);
        writer.start();
      }
      writer.putTable(table);
    } else if (writer != null) {
      // Include the transition out of Test mode so SysId receives the final state sample.
      writer.putTable(table);
      writer.end();
      writer = null;
    }
  }

  @Override
  public void end() {
    if (writer != null) {
      writer.end();
      writer = null;
    }
  }
}
