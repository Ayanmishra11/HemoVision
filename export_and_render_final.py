import win32com.client
import os
import fitz

ppt_app = win32com.client.Dispatch('PowerPoint.Application')
ppt_path = os.path.abspath('HEMOVISION_SIH_2026_FINAL.pptx')
pdf_path = os.path.abspath('HEMOVISION_SIH_2026_FINAL.pdf')

try:
    deck = ppt_app.Presentations.Open(ppt_path, WithWindow=False)
    deck.SaveAs(pdf_path, 32) # 32 is ppSaveAsPDF
    deck.Close()
    print('Successfully exported to PDF:', pdf_path)
except Exception as e:
    print('Error exporting to PDF:', e)
finally:
    ppt_app.Quit()

# Render all slides to PNG
os.makedirs('sih_final_preview', exist_ok=True)
doc = fitz.open(pdf_path)
print(f'PDF page count: {len(doc)}')
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    out_png = f'sih_final_preview/slide_{i+1}.png'
    pix.save(out_png)
    print(f'Saved preview: {out_png}')
