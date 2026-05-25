from datetime import timedelta
from xml.etree import ElementTree as ET

from bioio_czi import standard_metadata


def test_position_index_parses_image_scene_id() -> None:
    assert standard_metadata.position_index("Image:0") == 0
    assert standard_metadata.position_index("Image:12") == 12


def test_time_interval_from_metadata_xml() -> None:
    for xml in (
        """
        <Metadata>
          <Information>
            <Image>
              <Dimensions>
                <T>
                  <Positions>
                    <Interval>
                      <Increment>59.927</Increment>
                    </Interval>
                  </Positions>
                </T>
              </Dimensions>
            </Image>
          </Information>
        </Metadata>
        """,
        """
        <ImageDocument>
          <Metadata>
            <Information>
              <Image>
                <Dimensions>
                  <T>
                    <Positions>
                      <Interval>
                        <Increment>59.927</Increment>
                      </Interval>
                    </Positions>
                  </T>
                </Dimensions>
              </Image>
            </Information>
          </Metadata>
        </ImageDocument>
        """,
    ):
        metadata = ET.fromstring(xml)
        assert standard_metadata.time_interval(metadata) == timedelta(seconds=59.927)


def test_time_interval_missing_increment() -> None:
    metadata = ET.fromstring(
        """
        <Metadata>
          <Information>
            <Image>
              <Dimensions>
                <T>
                  <Positions>
                    <BinaryList>
                      <AttachmentName>TimeStamps</AttachmentName>
                    </BinaryList>
                  </Positions>
                </T>
              </Dimensions>
            </Image>
          </Information>
        </Metadata>
        """
    )

    assert standard_metadata.time_interval(metadata) is None
