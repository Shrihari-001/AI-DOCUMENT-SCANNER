
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def create_presentation(ppt_data):

    presentation = Presentation()

    # Title slide
    title_slide = presentation.slides.add_slide(
        presentation.slide_layouts[0]
    )

    title_slide.shapes.title.text = ppt_data["title"]
    title_slide.placeholders[1].text = (
        "Created by AI Document Scanner & Study Assistant"
    )

    # Content slides
    for slide_data in ppt_data["slides"]:

        slide = presentation.slides.add_slide(
            presentation.slide_layouts[1]
        )

        slide.shapes.title.text = slide_data["title"]

        content = slide.placeholders[1].text_frame
        content.clear()

        for index, point in enumerate(slide_data["points"]):

            if index == 0:
                paragraph = content.paragraphs[0]
            else:
                paragraph = content.add_paragraph()

            paragraph.text = str(point)
            paragraph.level = 0
            paragraph.font.size = Pt(22)
            paragraph.font.color.rgb = RGBColor(
                40, 40, 40
            )

    # Save presentation in memory
    from io import BytesIO

    ppt_buffer = BytesIO()

    presentation.save(ppt_buffer)

    ppt_buffer.seek(0)

    return ppt_buffer