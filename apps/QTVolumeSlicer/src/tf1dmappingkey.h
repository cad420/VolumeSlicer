#ifndef TF1DMAPPINGKEY_H
#define TF1DMAPPINGKEY_H

#include <QColor>

class TF1DMappingKey {
public:
    TF1DMappingKey(float i, const QColor& color)
		: intensity(i), colorL(color), colorR(color), split(false)
	{}
	~TF1DMappingKey() {}

    bool operator==(const TF1DMappingKey& key) {
		 return (intensity == key.intensity) && (split == key.split) &&
				(colorR == key.colorR) && (colorL == key.colorL);
	}
	bool operator!=(const TF1DMappingKey& key) { return !(*this == key); }

	void setColorL(const QColor& color) {
		colorL = color;
		if(!split)
			colorR = color;
	}

	QColor& getColorL() { return colorL; };

	void setColorR(const QColor& color) {
		colorR = color;
		if(!split)
			colorL = color;
	}

	QColor& getColorR() { return colorR; }

	void setAlphaL(float a) {
		colorL.setAlpha(static_cast<int>(a*255.f));
		if(!split)
			colorR.setAlpha(static_cast<int>(a*255.f));
	}

	float getAlphaL() { return colorL.alpha() / 255.f; }

	void setAlphaR(float a) {
		colorR.setAlpha(static_cast<int>(a * 255.f));
		if(!split)
			colorL.setAlpha(static_cast<int>(a*255.f));
	}

	float getAlphaR()const { return colorR.alpha() / 255.f; }

	void setSplit(bool split, bool useLeft = true) {
		if(this->split == split)
			return;
		if (!split) {
			//join colors
			if(useLeft)
				colorR = colorL;
			else
				colorL = colorR;
		}
		this->split = split;
	}

	bool isSplit() { return split; }

	void setIntensity(float i) { intensity = i; }

	float getIntensity()const { return intensity; }

private:
    float  intensity;	///< intensity at which the key is located
    QColor colorL;		///< color of the left part of the key
    QColor colorR;		///< color of the right part of the key
    bool   split;       ///< is the key split?
};

#endif